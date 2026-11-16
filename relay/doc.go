// Package relay serves immutable tensor blocks to workers so the coordinator does not have to.
//
// A relay is a dictionary from a block's SHA-256 (hex) to its bytes, with a memory budget,
// over HTTP. It knows nothing about tensors, tasks or versions: the coordinator decides which
// keys a task reads and when a key may be dropped. Because the key is the content hash, a
// read can never return a stale value and a write can never corrupt one.
//
// Endpoints (all need "Authorization: Bearer <token>"):
//
//	GET    /blocks/{key}      the bytes of one block (application/octet-stream), 404 if unknown
//	GET    /blocks?keys=a,b   several blocks in one Ygaraty binary frame, 404 listing any missing
//	POST   /blocks/get        the same, for JSON {"keys": [...]}: one fixed URL, so a browser's CORS
//	                          preflight is cached instead of paid per distinct key list
//	PUT    /blocks/{key}      store bytes whose SHA-256 is {key} (a repeat adds a reference); 400 on mismatch, 507 over budget
//	                          ?f64=1 also requires finite little-endian float64 values
//	                          (?f32=1: float32 values, ?bf16=1: bf16 values)
//	PUT    /blocks            several blocks in one frame (as GET /blocks answers), each header entry
//	                          {"key", "size", "check": "f64"|"f32"|"bf16"}; all are verified before any is
//	                          stored. One request (and one cached preflight) per result instead of one per block.
//	                          With "name" and "manifest" in the header, the frame is also published as a handoff
//	POST   /handoff/get       JSON {"names": [...], "keys": [...], "wait_ms": N}: wait (up to 60 s) until every name
//	                          is published, then one frame with {"manifests", "blocks"} and the bytes of the named
//	                          results' blocks and of the extra keys; 404 listing the names missing at the deadline
//	DELETE /blocks            JSON {"keys": [...]}: drop one reference per key; a block goes at zero
//	GET    /status            counters as JSON
//
// Both GETs answer gzip (Content-Encoding) when the request's Accept-Encoding allows it, the
// body is at least 1 KiB and gzip makes it smaller; clients that do not ask get the bytes as before.
// A PUT may be gzipped (Content-Encoding: gzip): its key and limit apply to the decoded bytes.
//
// A frame is a 4-byte little-endian header length, a JSON header padded with spaces to an
// 8-byte boundary, then the blocks' bytes back to back (ygaraty/transports/http/frames.py).
package relay
