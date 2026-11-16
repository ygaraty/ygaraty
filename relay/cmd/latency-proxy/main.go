// Command latency-proxy forwards TCP connections and delays every byte by half a round trip
// in each direction, preserving order. With -down-mbps and -up-mbps it also models one access
// link: every connection through this process shares one downstream and one upstream link of
// that rate, as a home connection does. It stands in for a network between remote workers and
// the coordinator or relay when traffic shaping (tc netem) is not available.
//
//	latency-proxy -forward 127.0.0.1:9001=127.0.0.1:8090 -forward 127.0.0.1:9002=127.0.0.1:8093 \
//	              -rtt-ms 50 -down-mbps 100 -up-mbps 20
//
// "Up" is the client's direction (bytes it sends), "down" the server's. -listen and -target
// remain as a shorthand for a single forward.
package main

import (
	"flag"
	"io"
	"log"
	"net"
	"strings"
	"sync"
	"time"
)

type chunk struct {
	data []byte
	due  time.Time
}

// link is one direction of an access link: bytes leave it one after another at `rate`.
type link struct {
	mu   sync.Mutex
	rate float64 // bytes per second; 0 is unlimited
	free time.Time
}

// transmit reserves the link for n bytes that may not leave before `due`, and returns when the
// last of them has left.
func (l *link) transmit(n int, due time.Time) time.Time {
	if l == nil || l.rate == 0 {
		return due
	}
	l.mu.Lock()
	start := due
	if l.free.After(start) {
		start = l.free
	}
	finish := start.Add(time.Duration(float64(n) / l.rate * float64(time.Second)))
	l.free = finish
	l.mu.Unlock()
	return finish
}

// pipe reads from src as fast as the queue allows, stamps each read, and a second goroutine
// writes it to dst once its delay has passed and its turn on the link has come.
func pipe(dst, src net.Conn, delay time.Duration, l *link) {
	queue := make(chan chunk, 64) // bounded, so a slow link pushes back on the sender
	go func() {
		defer close(queue)
		for {
			buf := make([]byte, 64<<10)
			n, err := src.Read(buf)
			if n > 0 {
				queue <- chunk{buf[:n], time.Now().Add(delay)}
			}
			if err != nil {
				return
			}
		}
	}()
	for c := range queue {
		if wait := time.Until(l.transmit(len(c.data), c.due)); wait > 0 {
			time.Sleep(wait)
		}
		if _, err := dst.Write(c.data); err != nil {
			break
		}
	}
	if tcp, ok := dst.(*net.TCPConn); ok {
		tcp.CloseWrite()
	} else {
		dst.Close()
	}
	io.Copy(io.Discard, src)
}

type forwards []string

func (f *forwards) String() string     { return strings.Join(*f, ",") }
func (f *forwards) Set(v string) error { *f = append(*f, v); return nil }

func main() {
	var pairs forwards
	flag.Var(&pairs, "forward", "listen=target, repeatable; every forward shares the access link")
	listen := flag.String("listen", "", "address to accept on (single forward)")
	target := flag.String("target", "", "address to forward to (single forward)")
	rtt := flag.Float64("rtt-ms", 50, "added round-trip time in milliseconds (half each way)")
	down := flag.Float64("down-mbps", 0, "downstream rate in megabits per second (server to client); 0 is unlimited")
	up := flag.Float64("up-mbps", 0, "upstream rate in megabits per second (client to server); 0 is unlimited")
	handshake := flag.Float64("handshake-rtts", 0, "round trips a new connection waits before its first byte moves: "+
		"the TCP handshake (1) and TLS 1.3 (1 more), which the local accept here does not cost; 0 keeps the old behaviour")
	flag.Parse()
	if *listen != "" {
		pairs = append(pairs, *listen+"="+*target)
	}
	if len(pairs) == 0 {
		log.Fatal("give -forward listen=target (or -listen and -target)")
	}
	delay := time.Duration(*rtt / 2 * float64(time.Millisecond))
	setup := time.Duration(*handshake * *rtt * float64(time.Millisecond))
	downLink, upLink := &link{rate: *down * 1e6 / 8}, &link{rate: *up * 1e6 / 8}
	var wg sync.WaitGroup
	for _, pair := range pairs {
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) != 2 {
			log.Fatalf("bad -forward %q", pair)
		}
		ln, err := net.Listen("tcp", parts[0])
		if err != nil {
			log.Fatal(err)
		}
		log.Printf("forwarding %s -> %s, %.1f ms round trip, down %.0f Mb/s, up %.0f Mb/s, %.0f handshake round trips", ln.Addr(), parts[1], *rtt, *down, *up, *handshake)
		wg.Add(1)
		go func(ln net.Listener, to string) {
			defer wg.Done()
			for {
				client, err := ln.Accept()
				if err != nil {
					log.Fatal(err)
				}
				go func() {
					time.Sleep(setup) // bytes the client already sent wait in the socket meanwhile
					server, err := net.Dial("tcp", to)
					if err != nil {
						log.Print(err)
						client.Close()
						return
					}
					done := make(chan struct{}, 2)
					go func() { pipe(server, client, delay, upLink); done <- struct{}{} }()
					go func() { pipe(client, server, delay, downLink); done <- struct{}{} }()
					<-done
					<-done
					client.Close()
					server.Close()
				}()
			}
		}(ln, parts[1])
	}
	wg.Wait()
}
