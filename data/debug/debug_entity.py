#!/usr/bin/env python3
import re
import html

# The key insight: #39; in raw text is MISSING the &
# The raw text has: "word #39;next" where space is BEFORE the #
# So we need: word 'next -> word'next

# Let me trace what the regex is doing
text = "doesn #39;t"
print(f"Original: {repr(text)}")

# The regex: (?<!&#)#(\d+);
# This matches # followed by digits followed by ;
# Let's trace step by step

# Step 1: The pattern looks for # that's NOT preceded by &
# In "doesn #39;t":
# - Position of #: right after the space
# - The char before # is space (not &)
# - So it MATCHES #39;
# - Capture group 1 = "39"
# - Replacement: &#\g<1>; -> &#39;

result = re.sub(r"(?<!&#)#(\d+);", r"&#\g<1>;", text)
print(f"After regex: {repr(result)}")
# This gives: "doesn &#39;t" - note the 't' is directly after ;

# After html.unescape:
result2 = html.unescape(result)
print(f"After unescape: {repr(result2)}")
# This gives: "doesn 't" - #39; became ' and t is right after

# But the input has SPACE before #39;
# So actually: "doesn #39;t" should become "doesn' t" NOT "doesn 't"

# Wait, let me re-read the spec example:
# "doesn #39;t" -> "doesn't"
# That means: space + #39; + t -> ' + t
# So the space before #39; should be removed
# And #39; becomes '
# And t stays as t
# Result: doesn + ' + t = doesn't

print("\n--- Testing different approaches ---")

# Approach 1: Current (wrong) - doesn't remove space before
text1 = "doesn #39;t"
result1 = html.unescape(re.sub(r"(?<!&#)#(\d+);", r"&#\g<1>;", text1))
print(f"Approach 1: {repr(result1)}")

# Approach 2: Remove space before, don't add after
text2 = "doesn #39;t"
# First fix: add & before #
text2 = re.sub(r"(?<!&#)#(\d+);", r"&#\g<1>;", text2)
print(f"After adding &: {repr(text2)}")
result2 = html.unescape(text2)
print(f"Approach 2: {repr(result2)}")

# Approach 3: What if the space is BETWEEN # and 39?
# Let's test "doesn # 39;t"
text3 = "doesn # 39;t"
result3 = html.unescape(re.sub(r"(?<!&#)#(\d+);", r"&#\g<1>;", text3))
print(f"If space between # and 39: {repr(result3)}")

# AH! I think the issue is the SPEC says:
# #39; without & should become &#39; then ' with space handling
# But maybe the expected behavior is: doesn't (contraction)
# Which means: doesn't = does not (contraction) with apostrophe

# Let me check: "doesn" + "'" + "t" = "doesn't"
# But the space is BEFORE #39;, not after
# So: "doesn #39;t" -> "doesn #39;t" (no change to space)
# Then: "doesn #39;t" -> "doesn &#39;t" (fix missing &)
# Then: "doesn &#39;t" -> "doesn 't" (html.unescape)
# Then: "doesn 't" -> "doesn't" (remove space between ' and t)

# But my code does:
# "doesn #39;t" -> "doesn &#39;t" (fix missing &)
# Then: "doesn &#39;t" -> "doesn 't" (html.unescape)
# Then: "doesn 't" -> "doesn 't" (no space to collapse - 't has no space)

# So the issue is: there IS a space before #39; originally ("doesn #39;t")
# But my output shows no space before the ' ("doesn 't")
# Let me re-check the regex...

print("\n--- More debugging ---")
text = "doesn #39;t"
print(f"Input: {repr(text)}")
print(f"Characters: {[c for c in text]}")
