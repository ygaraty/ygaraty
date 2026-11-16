package relay

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"sync"
	"sync/atomic"
	"time"
)

// Store holds blocks by content hash within a byte budget. Each block counts references:
// a PUT of a key already held adds one, a DELETE drops one, and the bytes go at zero. The
// coordinator pushes every block of every version it writes and releases per version, so
// identical bytes shared by two versions (zero tiles, say) survive the first release.
type Store struct {
	mu       sync.RWMutex
	blocks   map[string]*entry
	bytes    int64
	MaxBytes int64 // 0 means unlimited

	hits, misses, puts, duplicates, deletes, freed, servedBytes atomic.Int64
}

type entry struct {
	data []byte
	refs int
}

var (
	ErrBadKey       = errors.New("key must be 64 lowercase hex characters")
	ErrHashMismatch = errors.New("body does not hash to key")
	ErrOverBudget   = errors.New("store budget exceeded")
)

func NewStore(maxBytes int64) *Store {
	return &Store{blocks: make(map[string]*entry), MaxBytes: maxBytes}
}

func validKey(key string) bool {
	if len(key) != 64 {
		return false
	}
	for _, c := range key {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

// Put stores data under key, which must be its SHA-256. Storing a key again retains its bytes and adds one reference.
func (s *Store) Put(key string, data []byte) (created bool, err error) {
	if !validKey(key) {
		return false, ErrBadKey
	}
	sum := sha256.Sum256(data)
	if hex.EncodeToString(sum[:]) != key {
		return false, ErrHashMismatch
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.blocks[key]; ok {
		e.refs++
		s.duplicates.Add(1)
		return false, nil
	}
	if s.MaxBytes > 0 && s.bytes+int64(len(data)) > s.MaxBytes {
		return false, ErrOverBudget
	}
	s.blocks[key] = &entry{data: data, refs: 1}
	s.bytes += int64(len(data))
	s.puts.Add(1)
	return true, nil
}

// Get returns a block's bytes. The slice is immutable and must not be modified.
func (s *Store) Get(key string) ([]byte, bool) {
	s.mu.RLock()
	e, ok := s.blocks[key]
	s.mu.RUnlock()
	if !ok {
		s.misses.Add(1)
		return nil, false
	}
	s.hits.Add(1)
	s.servedBytes.Add(int64(len(e.data)))
	return e.data, true
}

// Delete drops one reference from each key. It returns how many keys were present and how
// many blocks were freed (reached zero references).
func (s *Store) Delete(keys []string) (present, freed int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, key := range keys {
		e, ok := s.blocks[key]
		if !ok {
			continue
		}
		present++
		e.refs--
		if e.refs <= 0 {
			delete(s.blocks, key)
			s.bytes -= int64(len(e.data))
			freed++
		}
	}
	s.deletes.Add(int64(present))
	s.freed.Add(int64(freed))
	return present, freed
}

// Status is the JSON body of GET /status.
type Status struct {
	Blocks      int   `json:"blocks"`
	Bytes       int64 `json:"bytes"`
	MaxBytes    int64 `json:"max_bytes"`
	Hits        int64 `json:"hits"`
	Misses      int64 `json:"misses"`
	Puts        int64 `json:"puts"`
	Duplicates  int64 `json:"duplicate_puts"`
	Deletes     int64 `json:"deletes"`
	Freed       int64 `json:"freed"`
	ServedBytes int64 `json:"served_bytes"`
	UptimeSec   int64 `json:"uptime_seconds"`
}

func (s *Store) Status(since time.Time) Status {
	s.mu.RLock()
	n, b := len(s.blocks), s.bytes
	s.mu.RUnlock()
	return Status{Blocks: n, Bytes: b, MaxBytes: s.MaxBytes, Hits: s.hits.Load(), Misses: s.misses.Load(),
		Puts: s.puts.Load(), Duplicates: s.duplicates.Load(), Deletes: s.deletes.Load(), Freed: s.freed.Load(),
		ServedBytes: s.servedBytes.Load(), UptimeSec: int64(time.Since(since).Seconds())}
}
