#!/usr/bin/env python3
"""Test script to verify trim_network_logs with postData conversion."""

import json
import sys
from pathlib import Path

# Add asi to path
sys.path.insert(0, str(Path(__file__).parent))

# Add webarena-verified to path (needed for network_event_utils)
# Path: agent-skill-induction/asi/../webarena-verified/src
webarena_src = Path(__file__).parent.parent.parent / "webarena-verified" / "src"
if webarena_src.exists():
    sys.path.insert(0, str(webarena_src))
else:
    print(f"⚠️  Warning: webarena-verified not found at {webarena_src}")

from trim_network_logs import trim_har_file


def test_trim_with_conversion():
    """Test trim_har_file with postData params to text conversion."""

    # Find a real HAR file from browsergym verification
    test_har_dir = Path(__file__).parent.parent.parent / "tmp" / "browsergym-har-verification"
    har_files = list(test_har_dir.glob("*/network.har"))

    if not har_files:
        print("⚠️  No HAR files found in browsergym-har-verification")
        return False

    test_har = har_files[0]
    print(f"Testing with: {test_har}")
    print(f"Original size: {test_har.stat().st_size:,} bytes")

    # Create output path
    output_har = test_har.parent / "network_trimmed_test.har"

    try:
        # Run trim with conversion
        stats = trim_har_file(test_har, output_har)

        print("\n✅ Trim completed successfully!")
        print(f"  Original entries: {stats['original_entries']}")
        print(f"  Trimmed entries: {stats['trimmed_entries']}")
        print(f"  Removed entries: {stats['removed_entries']}")
        print(f"  Request headers sanitized: {stats['request_headers_sanitized']}")
        print(f"  Response headers sanitized: {stats['response_headers_sanitized']}")
        print(f"  PostData converted: {stats['postdata_converted']}")  # ← New field!
        print(f"  Original size: {stats['original_size']:,} bytes")
        print(f"  Trimmed size: {stats['trimmed_size']:,} bytes")
        print(f"  Reduction: {stats['reduction_percent']:.1f}%")

        # Verify converted HAR has text in postData
        if stats['postdata_converted'] > 0:
            har_data = json.loads(output_har.read_text())
            found_converted = False

            for entry in har_data["log"]["entries"]:
                post_data = entry.get("request", {}).get("postData")
                if post_data and post_data.get("text"):
                    print(f"\n✅ Verified: Found converted postData with text:")
                    print(f"  URL: {entry['request']['url']}")
                    print(f"  Text length: {len(post_data['text'])} chars")
                    found_converted = True
                    break

            if not found_converted:
                print("\n⚠️  Warning: postdata_converted > 0 but no text found in output")

        # Cleanup
        output_har.unlink()
        print(f"\n🧹 Cleaned up test file: {output_har}")

        return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Testing trim_network_logs with postData conversion")
    print("=" * 60)

    success = test_trim_with_conversion()

    print("\n" + "=" * 60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Tests failed")
    print("=" * 60)

    sys.exit(0 if success else 1)
