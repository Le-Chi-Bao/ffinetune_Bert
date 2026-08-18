#!/usr/bin/env python3
import re
import html

# Test regex for numeric entity fix
text = 'doesn #39;t'
result = re.sub(r"(?<!&#)#(\d+);", r"&#\1;", text)
print(f"Test 1 - Input: {repr(text)}, Output: {repr(result)}")

text2 = 'earned #36;10'
result2 = re.sub(r"(?<!&#)#(\d+);", r"&#\1;", text2)
print(f"Test 2 - Input: {repr(text2)}, Output: {repr(result2)}")

# Test HTML unescape
print(f"\nHTML unescape of &#39;: {repr(html.unescape('&#39;'))}")
print(f"HTML unescape of &#36;: {repr(html.unescape('&#36;'))}")

# Full pipeline
def clean_test(text):
    text = re.sub(r"(?<!&#)#(\d+);", r"&#\1;", text)
    text = html.unescape(text)
    return text

print(f"\nFull pipeline test:")
print(f"'doesn #39;t' -> {repr(clean_test('doesn #39;t'))}")
print(f"'earned #36;10 million' -> {repr(clean_test('earned #36;10 million'))}")
