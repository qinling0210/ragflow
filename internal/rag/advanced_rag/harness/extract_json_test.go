package harness

import (
	"reflect"
	"testing"
)

// TestExtractJSONLenient mirrors Python action_session.extract_json's fallback
// through json.loads(strict=False) and json_repair.loads: models routinely emit
// trailing commas, single-quoted strings, bare NaN/Infinity and stray control
// characters. ExtractJSON should salvage these instead of dropping the whole
// candidate and moving to the next "{".

func eqJSONMap(t *testing.T, got any, want map[string]any) {
	t.Helper()
	g, ok := got.(map[string]any)
	if !ok {
		t.Fatalf("got %T %v, want map %v", got, got, want)
	}
	if !reflect.DeepEqual(g, want) {
		t.Fatalf("got %v, want %v", g, want)
	}
}

func TestExtractJSONLenient(t *testing.T) {
	cases := []struct {
		name string
		in   string
		want map[string]any
	}{
		{"valid passthrough", `{"a": 1}`, map[string]any{"a": float64(1)}},
		{"trailing comma", `{"a": 1,}`, map[string]any{"a": float64(1)}},
		{"single quotes", `{'a': 'b'}`, map[string]any{"a": "b"}},
		{"single quotes and trailing comma", `{'a': 1,}`, map[string]any{"a": float64(1)}},
		{"NaN literal", `{"x": NaN}`, map[string]any{"x": nil}},
		{"Infinity literal", `{"x": Infinity}`, map[string]any{"x": nil}},
		{"negative Infinity", `{"x": -Infinity}`, map[string]any{"x": nil}},
		{"control char in string", "{\"a\": \"x\x01y\"}", map[string]any{"a": "xy"}},
		{"prefixed text", `blah {"a": 1} blah`, map[string]any{"a": float64(1)}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			eqJSONMap(t, ExtractJSON(c.in), c.want)
		})
	}
}

func TestExtractJSONLenientUnrecoverable(t *testing.T) {
	// An object json_repair cannot salvage falls through to the next "{" and
	// eventually nil — it must not panic or hang.
	if v := ExtractJSON(`{not json at all`); v != nil {
		t.Fatalf("unrecoverable: got %v, want nil", v)
	}
	// Second object is salvaged after the first is hopeless.
	if v := ExtractJSON(`{"a":} {"b": 2}`); v != nil {
		eqJSONMap(t, v, map[string]any{"b": float64(2)})
	}
}
