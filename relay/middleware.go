package relay

import (
	"crypto/subtle"
	"net/http"
	"strings"
)

// cors answers a browser's preflight and marks every reply for an allowed page origin. The
// preflight carries no token (browsers send none), so it is answered before authentication;
// everything else still needs the bearer token. Replies vary by Origin, and a preflight may be
// cached for a day so one is paid per path, not per request.
func (rs *Server) cors(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		if origin != "" && rs.originAllowed(origin) {
			h := w.Header()
			h.Set("Access-Control-Allow-Origin", origin)
			h.Add("Vary", "Origin")
			if r.Method == http.MethodOptions {
				rs.preflights.Add(1)
				h.Set("Access-Control-Allow-Methods", "GET, POST, PUT, OPTIONS")
				h.Set("Access-Control-Allow-Headers", "Authorization, Content-Type, Accept-Encoding, Content-Encoding")
				h.Set("Access-Control-Max-Age", "86400")
				w.WriteHeader(http.StatusNoContent)
				return
			}
		} else if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		next.ServeHTTP(w, r)
	})
}

func (rs *Server) originAllowed(origin string) bool {
	for _, allowed := range rs.AllowedOrigins {
		if allowed == "*" || strings.EqualFold(allowed, origin) {
			return true
		}
	}
	return false
}

func (rs *Server) authenticate(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		token := strings.TrimPrefix(r.Header.Get("Authorization"), "Bearer ")
		if rs.Token == "" || subtle.ConstantTimeCompare([]byte(token), []byte(rs.Token)) != 1 {
			writeJSON(w, http.StatusUnauthorized, map[string]string{"error": "invalid access token"})
			return
		}
		next.ServeHTTP(w, r)
	})
}
