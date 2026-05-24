import json
import unittest

from app.api.sse import format_sse_event


class FormatSseEventTests(unittest.TestCase):
    def test_formats_named_event_with_json_data(self):
        payload = {"message": "hello", "count": 2}

        result = format_sse_event("trace", payload)

        self.assertEqual(
            result,
            f"event: trace\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n",
        )


if __name__ == "__main__":
    unittest.main()
