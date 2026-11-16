// Command relay runs a Ygaraty block relay.
//
//	relay -addr 127.0.0.1:8091 -token SECRET -max-mib 8192
//
// The token may also come from RELAY_TOKEN. Workers fetch task inputs here instead of from
// the coordinator; the coordinator pushes committed blocks and deletes released ones.
package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/ygaraty/ygaraty/relay"
)

func main() {
	addr := flag.String("addr", "127.0.0.1:8091", "listen address")
	token := flag.String("token", os.Getenv("RELAY_TOKEN"), "bearer token every request must carry (or RELAY_TOKEN)")
	maxMiB := flag.Int64("max-mib", 8192, "memory budget for blocks in MiB; 0 is unlimited")
	maxBlockMiB := flag.Int64("max-block-mib", 64, "largest block accepted in MiB")
	origins := flag.String("cors-origins", "", "comma-separated page origins allowed to call the relay from a browser (the coordinator's, e.g. https://coordinator.example); empty refuses browsers")
	flag.Parse()
	if *token == "" {
		log.Fatal("a token is required: -token or RELAY_TOKEN")
	}
	rs := &relay.Server{Store: relay.NewStore(*maxMiB << 20), Token: *token, MaxBlockBytes: *maxBlockMiB << 20}
	if *origins != "" {
		rs.AllowedOrigins = strings.Split(*origins, ",")
	}
	server := &http.Server{
		Addr:              *addr,
		Handler:           rs.Handler(),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	go func() {
		log.Printf("relay listening on %s, budget %d MiB", *addr, *maxMiB)
		if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatal(err)
		}
	}()
	stop := make(chan os.Signal, 1)
	signal.Notify(stop, os.Interrupt, syscall.SIGTERM)
	<-stop
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	log.Print("relay shutting down")
	server.Shutdown(ctx)
}
