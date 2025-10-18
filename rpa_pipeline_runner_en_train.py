import json
from pathlib import Path
import re
import pandas as pd
import requests
from key import KEY_LIST
import random

MODEL = "models/gemini-2.5-flash"
HEADERS = {"Content-Type": "application/json"}

current_key_index = random.randint(0, len(KEY_LIST) - 1)
used_keys = set()

def rotate_api_key():
    global current_key_index
    used_keys.add(current_key_index)
    print(f"🔁 Gemini key index {current_key_index} failed. Switching...")
    if len(used_keys) == len(KEY_LIST):
        print("❌ All Gemini API keys exhausted. Exiting.")
        exit(1)
    current_key_index = (current_key_index + 1) % len(KEY_LIST)

def get_url():
    return f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:generateContent?key={KEY_LIST[current_key_index]}"

def call_gemini(prompt: str):
    while True:
        body = {"contents": [{"parts": [{"text": prompt}]}]}
        response = requests.post(get_url(), headers=HEADERS, json=body)

        try:
            content = response.json()
            print(content.get('usageMetadata', {}))
            if 'candidates' in content:
                # Reset used_keys on success
                used_keys.clear()
                raw_text = content["candidates"][0]["content"]["parts"][0]["text"]
                if raw_text.strip().startswith("```json"):
                    raw_text = raw_text.strip().removeprefix("```json").removesuffix("```").strip()
                print(raw_text)
                return raw_text
            elif content.get("error", {}).get("code") == 429:
                print("❌ Gemini error:")
                print(json.dumps(content, indent=2, ensure_ascii=False))
                rotate_api_key()
            else:
                print("❌ Gemini error:")
                print(json.dumps(content, indent=2, ensure_ascii=False))
                rotate_api_key()
        except Exception as e:
            print("❌ Unexpected error:", e)
            try:
                print("📄 Full JSON content:")
                print(json.dumps(content, indent=2, ensure_ascii=False))
            except NameError:
                print("📄 Raw response text:")
                print(response.text[:2000])
            rotate_api_key()

def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def extract_json_from_text(text: str) -> str:
    text = text.strip()
    # Remove markdown code blocks and any leading ```python or ```
    text = re.sub(r"^```(?:python)?\n", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

def sanitize_numeric_quotes(s: str) -> str:
  """Remove single quotes around numeric units, color codes, css function values
  such as '16px', '2s', '800ms', '75%', '#ff0000', 'rgb(34,139,34)', '400x300px'.
  Keep single quotes for human-readable text like 'Submit', 'Roboto', URLs, file names.
  """
  patterns = [
    r"'(#[0-9A-Fa-f]{3,8})'",  # hex colors
    r"'((?:rgb|rgba|hsl|hsla)\([^']+\))'",  # css color functions
    r"'(\d+(?:\.\d+)?(?:px|em|rem|vw|vh|%)|\d+(?:ms|s))'",  # units and time
    r"'(\d{2,4}x\d{2,4}(?:px)?)'",  # dimensions like 400x300 or 400x300px
  ]
  for pat in patterns:
    s = re.sub(pat, r"\1", s)
  return s

def step0_create_task(filename: str):
    # Prepare folder & output file
    INPUT_DIR = Path("Input")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INPUT_DIR / (filename if filename.endswith(".txt") else f"{filename}.txt")
    prompt = """
You are a professional UI test designer.
Generate exactly 20 **English** UI test description strings.

Output format (must be followed exactly):
- Return ONLY a valid JSON array of exactly 20 strings.
- Use JSON double quotes for the array items (JSON requirement).
- Inside each string, do not include any double quotes; use only single quotes for quoted labels or literals.
- Do not include numbering, markdown, code fences, or any extra text outside the JSON array.

Strict constraints (must satisfy all):
- Output is a single JSON array, e.g., ["sentence 1", "sentence 2", ...]. Inside each string you may include 10–15 UI checks like the example below.
- Absolutely forbid vague words (e.g., 'suitable', 'fast', 'smooth', 'enough', ...). Always provide measurable values.
- Steps must be ordered and complete.
- Each sentence should describe many attributes of the same element (e.g., font size, color, border, ...).
- Each action should specify multiple attributes at once (style, layout, state, ...).
- Always provide concrete, measurable expected values with units or counts where applicable: exact pixels (e.g., 16px), RGB/HEX colors (e.g., rgb(34,139,34) or #228B22 or 'green'), time (e.g., 800ms, 2s), percentage (e.g., 75%), URL (full), viewport width (e.g., 360px), coordinates (x,y).
- Do not combine two unrelated actions in one sentence. If the same action applies to unrelated elements, split them into separate sentences.
- Inside each string, use only single quotes (') when quoting text/labels/links; do not use any double quotes (\") inside the content.
 - Quote usage policy:
   - Use single quotes ONLY when quoting human-readable text or identifiers such as: visible UI text content, link URLs, exact labels/aria-labels, font family names, file names (e.g., 'Submit', 'https://...', 'Roboto', 'hero_image.jpg').
   - Do NOT wrap numeric values, units, color codes, or CSS function values in single quotes. Disallowed forms include: '16px', '2s', '800ms', '75%', '#ff0000', 'rgb(34,139,34)', 'rgba(0,0,0,0.5)', 'hsl(0, 0%, 100%)', '400x300px'. These must appear without the extra single quotes.
   - Do not single-quote boolean-like words or states (visible/hidden, enabled/disabled, active/selected, true/false).

Coverage (ensure diversity across the 20 items):
- You are allowed to use ONLY the attributes below and MUST NOT use any attribute outside this list.
 - Style: color, background color, font size, font family, font weight, border (px), border radius, opacity, padding, margin
 - State & Interaction: visible/hidden, focused, enabled/disabled, active/selected
 - Position & Layout: x, y, width, height, alignment, left, right, top, bottom
 - Media: muted, fullscreen
 - Dynamics: transition, animation, scroll-top/scroll-left (scroll)
 - Text: text, placeholder, alt text, text align
 - Data: count, value, date
 - Image: source, rendered dimensions, natural dimensions, is loaded, load time, watermark
 - Accessibility: aria-label, label
Each description MUST include one or more attributes from the list above. Do not introduce any attributes outside the following whitelist.
Whitelist of permitted attributes:
1. color
2. background color
3. font size
4. font family
5. font weight
6. border (px)
7. border radius
8. opacity
9. padding
10. margin
11. text align
12. visible/hidden
13. focused
14. enabled/disabled
15. active/selected
16. x
17. y
18. width
19. height
20. alignment
21. left
22. right
23. top
24. bottom
25. muted
26. fullscreen
27. transition
28. animation
29. scroll
30. text
31. placeholder
32. alt text
33. count
34. value
35. date
36. source
37. rendered dimensions
38. natural dimensions
39. is loaded
40. load time
41. watermark
42. aria-label
43. label
44. shape

Output formatting (must follow exactly):
- Return ONLY a valid JSON array of exactly 20 strings.
- Use JSON double quotes for items.
- Inside each string, **do not** include any double quotes; use only single quotes for quoted labels/literals.
- Do not include numbering, markdown, code fences, or any other text outside the JSON array.
 - Enforce the quote usage policy above. Examples:
   - Allowed: font size 16px, color #ff0000, background rgb(34,139,34), timeout 800ms.
   - Disallowed (single-quoted values): '16px', '#ff0000', 'rgb(34,139,34)', '800ms'.

Synonym usage requirements (for linguistic diversity while preserving semantics):
- Do NOT change the original meaning of any attribute (e.g., 'background color' vs 'background hue' must still mean the same CSS property).
- Across the 20 descriptions, alternate natural synonyms for the same attribute type to increase language variety (e.g., 'font size' / 'text size' / 'type size').
- In the model's JSON output, represent values purely as text strings; no reverse mapping is required.
- The goal is to teach the model that different phrases can describe the same UI characteristic without altering the underlying attribute.

Use **English** for everything (element names, actions, descriptions).

To encourage generalization, you may use natural synonyms for CSS properties while keeping the intended meaning and allowing unambiguous mapping back to the whitelist.
Use **English** for everything (element names, actions, descriptions).

Valid example of style (do NOT copy):
"Open 'https://www.shopdemo.com/product/123'. Verify the product image has source 'hero_image.jpg' and rendered size 400x300px. Ensure the image's alt text contains 'Genuine Shoes'. Check exactly 6 thumbnails are displayed below, centered, each 120px wide and 80px high. Confirm the main heading displays the exact text 'Product Title' with font size 18px and color #000000. Verify the 'Buy now' button is visible and becomes selected with background #ff0000. Confirm the button has border radius 6px, perform a scroll, and verify the header remains sticky during scrolling. Check that the language label has aria-label set to English."
Return 20 such strings inside one JSON array. Return the JSON array only.
"""
    all_sentences = []
    for _ in range(5):  # 5 batches -> ~100 strings
        raw = call_gemini(prompt)
        text = extract_json_from_text(raw) if raw else ""
        sentences = []
        # 1) try to parse as JSON array
        try:
            data = json.loads(text)
            if isinstance(data, list):
                sentences = [str(s) for s in data]
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            pass
        # 2) fallback: split by lines
        if not sentences:
            sentences = [ln.strip().strip(",") for ln in text.splitlines() if ln.strip()]
    # Normalize: replace inner " with ' ; sanitize numeric/color quotes; wrap each sentence with "
    for s in sentences:
      s = s.replace('"', "'").strip()
      s = sanitize_numeric_quotes(s)
      if not (s.startswith('"') and s.endswith('"')):
        s = f'"{s}"'
      all_sentences.append(s)

    # Write the file: comma-separated strings, each kept inside quotes
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",\n".join(all_sentences))
    print(f"✅ Generated {len(all_sentences)} sentences -> {out_path}")

def step1_analyze_task(input_file, batch_size=120):
    full_text = load_text(input_file)
    main_tasks = re.findall(r'"(.*?)"', full_text, re.DOTALL)
    print(len(main_tasks))
    task_list = []
    task_trace = []

    for start in range(0, len(main_tasks), batch_size):
        batch = main_tasks[start:start + batch_size]
        prompt = f"""You are a professional UI test-case atomizer.
Your job is to convert each natural-language test description into a list of clear, atomic UI actions.
Follow these strict rules:
- Output must be a JSON array of arrays.
- Group related attributes of the same UI element in the same state into a single atomic task.
  - This includes all visual, layout, formatting, style, alignment, and content attributes (e.g., font size, color, background, border radius, alignment).
  - Do not split alt text, watermark, natural vs rendered dimensions, sharpness/compression — if they refer to the same image in its normal state.
  - Do not split color, formatting, alignment, font, padding, size, position, or comparisons with another element — if they belong to the same element in the same context.
  - (Example: The 'Add to cart' button has color '#ff9900', border radius '4px' and background changes to '#e68a00' when selected)
  - Only split tasks if:
    - Attributes refer to different states (e.g., hover vs normal).
    - Attributes refer to different elements.
    - A user action followed by a verification is present (e.g., click then verify; locate then verify; locate then type), or an explicit user action is required in between.
    - An instruction includes multiple actions (e.g., "Locate 'Email' field and type 'invalid-email'" → split into 2 tasks).
    - Behavior differs between device types (mobile vs desktop).
    - The action requires multiple steps (e.g., "Log in with account 'patient001' and password 'Health@2023'" → split into 2 tasks).
- Each item in the array is a single atomic test instruction string.
- Do not invent steps that are not described in the main task.
Return ONLY the JSON array. No explanation, no markdown.
Quote usage policy for atomic instruction text:
- Use single quotes ONLY for human-readable text, exact labels, aria-labels, URLs, and font family/file names (e.g., 'Submit', 'https://...', 'Roboto', 'hero_image.jpg').
- NEVER quote numeric values, CSS units, color codes, or CSS function values. Avoid forms like '16px', '2s', '800ms', '75%', '#ff0000', 'rgb(34,139,34)'. Use them without quotes.
Examples:
- Requirement:
[
"Open 'https://www.netflix.com'. Click the 'Sign In' button at the top-right corner (50px from top and 30px from right). Enter email 'testuser@example.com' and password 'Test@1234'. Submit the login form. Verify that after submission, the user is redirected to 'https://www.netflix.com/browse' within 3 seconds and the page shows at least 5 personalized movie thumbnails; ensure all input fields use Roboto 16px, padding 12px, margin-bottom 16px. The 'Sign In' button has background #e50914, text color #ffffff, border radius 4px, is responsive at viewport width 360px, and all visible elements meet WCAG contrast ratio of at least 4.5:1.",
"Open 'https://www.amazon.com'. Search 'Bluetooth Speakers'; verify search completes under 2 seconds and shows at least 10 products with titles containing 'Bluetooth Speaker'. Confirm each product card image width is exactly 150px; price font size is 18px; and the 'Add to Cart' button has color #ff9900, border radius 4px, and hover background changes to #e68a00.",
"Open 'https://shop.example.com/item/P123'; verify the main product image '#main-product-image' loads within 800ms, alt text contains 'Premium Leather Wallet', no watermark, natural size 1600x1200 px, rendered size 400x300 px (±4px), sharpness ≥ 0.85, compression artifacts ≤ 0.1, and the image is positioned to the left of the price block '.price' with a minimum gap of 24px (±2px)."
]
- Response:
[
  [
"Open 'https://www.netflix.com'",
"Locate the 'Sign In' button at top-right (top offset 50px, right offset 30px)",
"Click the 'Sign In' button",
"Type 'testuser@example.com' into the email input field",
"Type 'Test@1234' into the password input field",
"Submit the login form",
"Verify redirection to 'https://www.netflix.com/browse' within 3 seconds",
"Verify at least 5 personalized movie thumbnails are visible",
"Verify all input fields have font family Roboto 16px, padding 12px, margin-bottom 16px",
"Verify the 'Sign In' button background #e50914, text color #ffffff, border radius 4px",
"Verify the 'Sign In' button is responsive at viewport width 360px",
"Verify all visible elements meet contrast ratio ≥ 4.5:1"
  ],
  [
"Open 'https://www.amazon.com'",
"Search 'Bluetooth Speakers'",
"Verify the search completes under 2 seconds",
"Verify search results show at least 10 products with title containing 'Bluetooth Speaker'",
"Verify each product card main image width is exactly 150px",
"Verify product price font size is 18px",
"Verify the 'Add to Cart' button has color #ff9900 and border radius 4px",
"Verify the 'Add to Cart' button hover background changes to #e68a00"
  ],
  [
"Open 'https://shop.example.com/item/P123'",
"Verify '#main-product-image' loads within 800ms, alt text contains 'Premium Leather Wallet', no watermark, natural size 1600x1200 px, rendered size 400x300 px (±4px), sharpness ≥ 0.85, compression artifacts ≤ 0.1, positioned left of '.price' with ≥24px (±2px) gap"
  ]
]
Now create the JSON from the following requirements (keep the same order):

- Requirement:
{chr(10).join([f'{i+1}. \"{t}\"' for i, t in enumerate(batch)])}

Return ONLY the JSON array of arrays in the same order.
"""
        retries = 3
        for attempt in range(retries):
            try:
                print("Call gemini step 1")
                response = call_gemini(prompt)
                clean_json = extract_json_from_text(response)
                result = json.loads(clean_json)
                if len(batch) == 1 and isinstance(result, list) and all(isinstance(x, str) for x in result):
                    result = [result]
                # Validate
                if not isinstance(result, list) or len(result) != len(batch) or any(not isinstance(x, list) for x in result):
                    raise ValueError("Model did not return a valid array of arrays with matching length.")
                for req_text, subtasks in zip(batch, result):
                    task_list.append(subtasks)
                    task_trace.append((req_text, subtasks))
                break
            except Exception as e:
                print(f"❌ Error (attempt {attempt + 1}/3): {e}")
                print(f"⚠️ Raw response: {response}")
                if attempt == retries - 1:
                    for req_text in batch:
                        task_list.append([])
                        task_trace.append((req_text, []))
                    print("⚠️ Continued with empty results for this batch.")

    return task_list, task_trace

def step2_generate_steps(subtask_groups):
    flat_tasks = [t for group in subtask_groups for t in group]
    print(f"Total tasks: {len(flat_tasks)}")
    number = 20
    grouped_tasks = [flat_tasks[i:i+number] for i in range(0, len(flat_tasks), number)]
    print(f"Grouped tasks: {len(grouped_tasks)}")
    all_step_groups = []
    step_trace = []
    step_map = []
    for subtask_list in grouped_tasks:    
        prompt = f"""You are a professional UI test step generator.
Your task is to convert the following **English** atomic test instruction list into a list of executable UI test steps in JSON format.

Instruction: "{subtask_list}"

Each step must follow this JSON structure:
{{
  "action": "action to perform (e.g. click, type, hover, locate, verify, etc.)",
  "selector": "CSS selector or a natural description",
  "value": "string value to type or expect, or empty string if not applicable",
  "expected": {{
    // This object may contain multiple properties,
    // based only on what the instruction requires.
    // Each key is the name of a UI property (e.g. color, font, position).
    // Each value is the expected value.
  }}
}}

Explanation of the actions:
- click, type, hover, select, submit, search, goto, etc.
- locate: find UI elements based on text, position (near/above), or attributes (class, ID, etc.).
- verify: validate properties like color, alignment, text, count, image presence, OCR output, size, position, visibility, etc.

Important:
- Always return output in JSON with English keys.
- All fields (`action`, `selector`, `value`, `expected`) **must always be present**.
- If a field has no value, use:
  - `""` for empty strings
  - `{{}}` for empty object (when `expected` is not needed)
- Do not omit any field from the JSON step.
- Do not include any property in `expected` if it is not clearly mentioned in the instruction.
- `expected` must be a valid object (`{{}}`), and can contain (examples only, include **only** if present in input):
  - 'text', 'placeholder', 'language'
  - 'position', 'x', 'y', 'width', 'height'
  - 'overflow', 'occluded'
  - 'text-align', 'line-height', 'vertical-align'
  - 'align-items', 'justify-content', 'align-content', 'align-self', 'gap'
  - 'opacity'
  - 'visibility', 'enabled', 'focused', 'hovered', 'active'
  - dynamics: 'movement', 'scroll', 'transition', 'animation', 'focus', 'navigation', 'loading', 'progress-value'
  - image-only: 'src', 'alt-text', 'is-loaded', 'load-time', 'sharpness-score-gte', 'compression-artifacts-lte', 'rendered-dimensions', 'natural-dimensions', 'viewport-dimensions', 'watermark'

Quote usage policy for JSON step values:
- Only quote string literals that are human-readable text, URLs, labels/aria-labels, font family names, or file names. Example: 'Submit', 'https://...', 'Roboto', 'hero_image.jpg'.
- Do NOT put quotes around numeric values, units, or color codes in `expected`. Examples (correct): width: 150px, timeout: 800ms, color: #ff9900, background-color: rgb(34,139,34).
- Avoid forms like '16px', '2s', '800ms', '#ff9900', 'rgb(34,139,34)' in `expected`.

Selenium + ChromeDriver constraints (Chrome computed styles):
1. Only include CSS properties retrievable via `window.getComputedStyle(element)`.
2. Property names must be lowercase, hyphen-separated (kebab-case). Examples: background-color, font-weight, border-radius, box-shadow.
3. Do NOT use camelCase or PascalCase.

Example 1:
- Task list:
[
  "Open 'https://www.amazon.com'",
  "Search 'Bluetooth Speakers'",
  "Verify search completes under 2 seconds",
  "Verify at least 10 product results with title containing 'Bluetooth Speaker'",
  "Verify each product card main image width is exactly 150px",
  "Verify the 'Add to Cart' button has color #ff9900 and border radius 4px, positioned at x=120px and y=300px, with width 200px and height 50px",
  "Verify the 'Add to Cart' button hover background changes to #e68a00"
]
- Response:
[
  {{
    "action": "goto",
    "selector": "",
    "value": "https://www.amazon.com",
    "expected": {{
      "url": "https://www.amazon.com"
    }}
  }},
  {{
    "action": "search",
    "selector": "'Search' input field",
    "value": "Bluetooth Speakers",
    "expected": {{
    }}
  }},
  {{
    "action": "verify",
    "selector": "search results",
    "value": "",
    "expected": {{
      "visibility": "visible",
      "timeout": 2000
    }}
  }},
  {{
    "action": "verify",
    "selector": ".search-results .product-title",
    "value": "",
    "expected": {{
      "count": 10,
      "text-content": "Bluetooth Speaker"
    }}
  }},
  {{
    "action": "verify",
    "selector": "'.product-card .main-image'",
    "value": "",
    "expected": {{
      "width": "150px"
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Add to Cart' button",
    "value": "",
    "expected": {{
      "text-content": "Add to Cart",
      "color": "#ff9900",
      "border-radius": "4px",
      "x": "120px",
      "y": "300px",
      "width": "200px",
      "height": "50px"
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Add to Cart' button",
    "value": "",
    "expected": {{
      "hovered": true,
      "background-color": "#e68a00"
    }}
  }}
]

Example 2:
- Task list:
[
  "Locate the 'Sign In' button at top-right (top 50px, right 30px)",
  "Click the 'Sign In' button",
  "Type 'testuser@example.com' into the email field",
  "Type 'Test@1234' into the password field",
  "Submit the login form",
  "Verify redirection to 'https://www.netflix.com/browse' within 3 seconds",
  "Verify at least 5 personalized movie thumbnails are visible",
  "Verify inputs use Roboto 16px, padding 12px, margin-bottom 16px",
  "Verify the 'Sign In' button background #e50914, text color #ffffff, border radius 4px",
  "Verify the 'Sign In' button is responsive at viewport width 360px",
  "Verify all visible elements meet contrast ratio ≥ 4.5:1"
]
- Response:
[
  {{
    "action": "locate",
    "selector": "'Sign In' button",
    "value": "",
    "expected": {{
      "text-content": "Sign In",
      "top": "50px",
      "right": "30px"
    }}
  }},
  {{
    "action": "click",
    "selector": "'Sign In' button",
    "value": "",
    "expected": {{
    }}
  }},
  {{
    "action": "type",
    "selector": "email input field",
    "value": "testuser@example.com",
    "expected": {{
    }}
  }},
  {{
    "action": "type",
    "selector": "password input field",
    "value": "Test@1234",
    "expected": {{
    }}
  }},
  {{
    "action": "submit",
    "selector": "login form",
    "value": "",
    "expected": {{
    }}
  }},
  {{
    "action": "verify",
    "selector": "",
    "value": "",
    "expected": {{
      "url": "https://www.netflix.com/browse",
      "timeout": 3000
    }}
  }},
  {{
    "action": "verify",
    "selector": "personalized movie thumbnails",
    "value": "",
    "expected": {{
      "count": 5
    }}
  }},
  {{
    "action": "verify",
    "selector": "input",
    "value": "",
    "expected": {{
      "font-family": "Roboto",
      "font-size": "16px",
      "padding-top": "12px",
      "padding-right": "12px",
      "padding-bottom": "12px",
      "padding-left": "12px",
      "margin-bottom": "16px"
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Sign In' button",
    "value": "",
    "expected": {{
      "text-content": "Sign In",
      "background-color": "#e50914",
      "color": "#ffffff",
      "border-radius": "4px"
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Sign In' button",
    "value": "",
    "expected": {{
      "viewport-width": "360px",
      "visibility": "visible",
      "enabled": true
    }}
  }},
  {{
    "action": "verify",
    "selector": "all visible elements",
    "value": "",
    "expected": {{
      "contrast-ratio": "at least 4.5:1"
    }}
  }}
]

Example 3:
- Task list:
[
  "Open 'https://shop.example.com/item/P123'",
  "Verify main product image '#main-product-image' loads within 800ms, alt text contains 'Premium Leather Wallet', no watermark, natural size 1600x1200, rendered size 400x300, sharpness ≥ 0.85, compression artifacts ≤ 0.1"
]
- Response:
[
  {{
    "action": "goto",
    "selector": "",
    "value": "https://shop.example.com/item/P123",
    "expected": {{
      "url": "https://shop.example.com/item/P123"
    }}
  }},
  {{
    "action": "verify",
    "selector": "#main-product-image",
    "value": "",
    "expected": {{
      "is-loaded": true,
      "load-time": 800,
      "alt-text": "Premium Leather Wallet",
      "watermark": false,
      "natural-width": 1600,
      "natural-height": 1200,
      "rendered-width": 400,
      "rendered-height": 300,
      "sharpness-score-gte": 0.85,
      "compression-artifacts-lte": 0.1
    }}
  }}
]

Notes:
- Each atomic instruction returns exactly ONE step (do NOT split an atomic instruction into multiple steps).
- The selector can be a CSS selector or a natural-language description.
- UI may be implemented using any frontend stack (TailwindCSS, Bootstrap, plain HTML).
- Return only a JSON array. No explanation. No markdown.

Now generate the JSON array of steps for the instruction above.
"""
        retries = 3
        for attempt in range(retries): 
            try:
                print("Call gemini step 2")
                response = call_gemini(prompt)
                clean = extract_json_from_text(response)
                steps = json.loads(clean)
                # Basic check: same length as subtask_list
                if not isinstance(steps, list) or len(steps) != len(subtask_list):
                    raise ValueError("Model did not return a JSON array with the same length as input.")
                # Ensure each object has 4 keys with defaults
                normalized = []
                for idx, st in enumerate(steps):
                    if not isinstance(st, dict):
                        raise ValueError(f"Step at index {idx} is not an object.")
                    action = st.get("action", "")
                    selector = st.get("selector", "")
                    value = st.get("value", "")
                    expected = st.get("expected", {})
                    if expected is None or not isinstance(expected, dict):
                        expected = {}
                    step_obj = {
                        "action": action if isinstance(action, str) else "",
                        "selector": selector if isinstance(selector, str) else "",
                        "value": value if isinstance(value, str) else "",
                        "expected": expected
                    }
                    normalized.append(step_obj)
                    step_map.append((subtask_list[idx], [step_obj]))

                all_step_groups.append(normalized)
                break

            except Exception as e:
                print(f"❌ Error (attempt {attempt + 1}/3): {e}")
                print(f"⚠️ Raw response: {response}")
                if attempt == retries - 1:
                    # Fallback: create empty steps to keep pipeline consistent
                    fallback = [{
                        "action": "",
                        "selector": "",
                        "value": "",
                        "expected": {}
                    } for _ in subtask_list]
                    all_step_groups.append(fallback)
                    for idx, st in enumerate(subtask_list):
                        step_map.append((st, [fallback[idx]]))
    # Reconstruct to match original group structure
    step_trace = step_map
    step_groups_flat = [steps for _, steps in step_map]

    reconstructed = []
    idx = 0
    for group in subtask_groups:
        size = len(group)
        reconstructed.append([step_groups_flat[idx + j][0] for j in range(size)])
        idx += size

    return reconstructed, step_trace

def save_excel_summary(filename, task_trace, step_groups):
    rows = []
    for idx, (main_task, subtasks) in enumerate(task_trace):
        step_group = step_groups[idx] if idx < len(step_groups) else []
        subtask_json = json.dumps(subtasks, ensure_ascii=False)
        step_json = json.dumps(step_group, ensure_ascii=False)
        rows.append({
            "Main Task": main_task,
            "Sub Tasks": subtask_json,
            "Steps": step_json
        })
    df = pd.DataFrame(rows)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(filename, index=False)

def main():
    inputName = input("Enter the base name for input files: ")

    INPUT_DIR = Path("Input")
    TASK_DIR = Path("JSONtask")
    STEP_DIR = Path("JSONwStep")
    REPORT_DIR = Path("Report")
    for d in [TASK_DIR, STEP_DIR, REPORT_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    numFile = 20
    print(f"Generate {numFile} files")
    for i in range(numFile):
        filename = f"{inputName}_{i}.txt"
        if not (INPUT_DIR / filename).exists():
            print(f"Creating example file: {filename}")
            step0_create_task(filename)
        else:
            print(f"File already exists: {filename}")
        case_id = Path(filename).stem
        try:
            print(f"\\n▶️ Running pipeline for: {case_id}")
            task_file = INPUT_DIR / f"{case_id}.txt"
            task_json_file = TASK_DIR / f"{case_id}.task.json"
            task_list, task_trace = step1_analyze_task(task_file)
            save_json(task_json_file, task_list)
            print("\\n Step 1 success \\n")  

            step_json_file = STEP_DIR / f"{case_id}.step.json"
            step_groups, step_trace = step2_generate_steps(task_list)
            save_json(step_json_file, step_groups)
            print("\\n Step 2 success \\n")

            report_file = REPORT_DIR / f"{case_id}.summary.xlsx"
            save_excel_summary(report_file, task_trace, step_groups)
            print(f"✅ Excel saved to: {report_file}")
        except Exception as e:
            print(f"❌ Error in {case_id}: {e}")

if __name__ == "__main__":
    main()
