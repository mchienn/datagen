"""
RPA Pipeline Runner for Train/Test Data Generation

ATTRIBUTE GROUPS DEFINITION (70-30% Split):
==========================================

STYLES (16 attributes): 70% train (11) + 30% test (5)
- Train: color, background color, font family, font size, font weight, text alignment, opacity, border, padding, margin, gap
- Test: fill, visited state color, line height, border radius, box shadow

STATES & INTERACTIVITY (7 attributes): 70% train (5) + 30% test (2)  
- Train: visible/hidden, enabled/disabled, focused, hovered, active/selected
- Test: clickable, cursor

POSITION & LAYOUT (17 attributes): 70% train (12) + 30% test (5)
- Train: x/y, width, height, alignment (absolute/relative), gap, aspect ratio, gap Px, offset, tolerance Px, z index, occluded, overflow
- Test: transform, clipping, blurring, left/right, top/bottom

CONTENT (5 attributes): 3-2 split
- Train: text, placeholder, language
- Test: exists, present

MEDIA (4 attributes): 3-1 split  
- Train: muted, playing, fullscreen
- Test: sourceType

DYNAMICS (12 attributes): 70% train (8) + 30% test (4)
- Train: transition, animation, movement, parallax, progress direction, scroll, loading, progress value
- Test: duration, auto advance, auto play, timeout

TEXT (6 attributes): 70% train (4) + 30% test (2)
- Train: text, validation, placeholder, allowed characters
- Test: language, alt text

DATA (5 attributes): 3-2 split
- Train: count, value, time value
- Test: date, currency symbol

IMAGE (10 attributes): 70% train (7) + 30% test (3)
- Train: source, is loaded, load time, sharpness score GTE, compression artifacts LTE, watermark, natural dimensions
- Test: rendered dimensions, viewport dimensions, video/audio

ACCESSIBILITY (7 attributes): 70% train (5) + 30% test (2)
- Train: activatable by keyboard, keyboard focusable, keyboard navigable, focus trapping, label
- Test: aria-label, contrast ratio

REPETITIVE & HIERARCHICAL (2 attributes): 1-1 split
- Train: list/table validation
- Test: nested elements check

COMPLEX VISUAL (3 attributes): 2-1 split
- Train: shape validation, image composition
- Test: graphics/charts validation

MORE (3 attributes): 2-1 split  
- Train: flowchart validation, browser compatibility
- Test: multiple attribute comparison
"""

import json
from pathlib import Path
import re
import pandas as pd
import requests
# from key import KEY_LIST
import random

KEY_LIST = [
    "AIzaSyDLsflIe6roJheFgYt2-yZNhWkefSDplxc",
    "AIzaSyCe1t0ZKd72Cbl5w_LUv8s9GES_zGxRJRc", 
    "AIzaSyCsL1qL0hDsPoEMlhNwpi-grZDE_qMef64",
    "AIzaSyDX6K9vXFqeF5yvFVCh4TXUyJvDPOezu2w",
    "AIzaSyANRpsExuz-iG2PJJ4qDs09ffPDVGz_xjk",
    "AIzaSyB41FVgW58dakFYYMyRFK6ECO1btw0YJ34",
    "AIzaSyD7W0WmJgJq20j5uqsQhPkd5rBN7fgI7D0",
    "AIzaSyDS0FU4TtXAyZo_5LVVgYO6r0VwiQCkSyk",
    "AIzaSyASJhbOMdoDskCw0DK064m6MRcYP3RzOek",
    "AIzaSyBeE2zDsYSevNjT2jAjDTAV78xOJgHkyKk",
    "AIzaSyCebrHwLcUfp50PIjc07Ln3mPuQNa7mqmY",
    "AIzaSyBxq1XVTElXyWYhqBunCAXBaGXAhHCr9hg",
    "AIzaSyDR0iAaefnM77-r82LQMewDCdaxx1Z5eu4",
    "AIzaSyB4-tYIo1th7F2yJJG4NAhW6-u2fcswXRY", 
    "AIzaSyDhWSf8YfEjIxH-dZgU7anIiT9HKkRVmJg",
    "AIzaSyBZ4VzwuVoys9ih-56kpFB1B-rW5eaZhXs",
    "AIzaSyCjc9t158aikfIixvwkV-2_dDOG3TXqWKg",
    "AIzaSyBywyWgvxcQn9aAQfjQ6U-st_tXROGOEEk",
    "AIzaSyBy4fptcpyNCSE5M9UoqpSsy_Tw-KU4dLk",
    "AIzaSyAAq6AZBZA7pGBd6SRfdQPaYBIQ-f5mjos",
    "AIzaSyAA1vPX8HRezaVel9RXqRTa9NEnJ831NeY",
    "AIzaSyDCb5MYyS5ucWvg_vBEbORMzRoC18T2so8",
    "AIzaSyA-hFzf05Io-4bEn-siACb3ZXnCxMUhkcg",
    "AIzaSyAkv4KdSRsUB_qsYvrS8GtZQkZBannFUbY"
    ]
MODEL = "models/gemini-2.5-flash"
HEADERS = {"Content-Type": "application/json"}

current_key_index = random.randint(0, len(KEY_LIST) - 1)
used_keys = set()

# Định nghĩa attributes cho train và test theo phân chia 70-30% và 3-2, 2-1, 1-1
TRAIN_ATTRIBUTES = [
    # Styles (11/16)
    "color", "background color", "font family", "font size", "font weight", 
    "text alignment", "opacity", "border", "padding", "margin", "gap",
    # States & Interactivity (5/7)
    "visible/hidden", "enabled/disabled", "focused", "hovered", "active/selected",
    # Position & Layout (12/17)
    "x/y coordinates", "width", "height", "alignment (absolute/relative)", 
    "gap", "aspect ratio", "gap Px", "offset", "tolerance Px", "z index", "occluded", "overflow",
    # Content (3/5)
    "text", "placeholder", "language",
    # Media (3/4)
    "muted", "playing", "fullscreen",
    # Dynamics (8/12)
    "transition", "animation", "movement", "parallax", "progress direction", 
    "scroll", "loading", "progress value",
    # Text (4/6)
    "text validation", "placeholder text", "allowed characters", "text content",
    # Data (3/5)
    "count", "value", "time value",
    # Image (7/10)
    "source", "is loaded", "load time", "sharpness score GTE", 
    "compression artifacts LTE", "watermark", "natural dimensions",
    # Accessibility (5/7)
    "activatable by keyboard", "keyboard focusable", "keyboard navigable", 
    "focus trapping", "label",
    # Repetitive & Hierarchical (1/2)
    "list/table validation",
    # Complex Visual (2/3)
    "shape validation", "image composition",
    # More (2/3)
    "flowchart validation", "browser compatibility"
]

TEST_ATTRIBUTES = [
    # Styles (5/16)
    "fill", "visited state color", "line height", "border radius", "box shadow",
    # States & Interactivity (2/7)
    "clickable", "cursor",
    # Position & Layout (5/17)
    "transform", "clipping", "blurring", "left/right", "top/bottom",
    # Content (2/5)
    "exists", "present",
    # Media (1/4)
    "source type",
    # Dynamics (4/12)
    "duration", "auto advance", "auto play", "timeout",
    # Text (2/6)
    "language detection", "alt text",
    # Data (2/5)
    "date", "currency symbol",
    # Image (3/10)
    "rendered dimensions", "viewport dimensions", "video/audio",
    # Accessibility (2/7)
    "aria-label", "contrast ratio",
    # Repetitive & Hierarchical (1/2)
    "nested elements check",
    # Complex Visual (1/3)
    "graphics/charts validation",
    # More (1/3)
    "multiple attribute comparison"
]

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
            print(content['usageMetadata'])
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
                # Nếu đã có content JSON, in đầy đủ
                print("📄 Full JSON content:")
                print(json.dumps(content, indent=2, ensure_ascii=False))
            except NameError:
                # Nếu chưa parse được content thì in raw text
                print("📄 Raw response text:")
                print(response.text[:2000])  # Giới hạn 2000 ký tự tránh log quá dài
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
    text = re.sub(r"^```(?:python)?\\n", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

def step0_create_train_task(filename: str):
    """Create task using TRAIN_ATTRIBUTES"""
    INPUT_DIR = Path("Input_Train")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INPUT_DIR / (filename if filename.endswith(".txt") else f"{filename}.txt")
    
    prompt = f"""
You are a professional UI tester.
Generate exactly 20 different UI test descriptions in English.

Output format (must be followed exactly):
- Return ONLY a valid JSON array of exactly 20 strings.
- Use JSON double quotes for the array items (JSON requirement).
- Inside each string, do not include any double quotes; use only single quotes for quoted labels or literals.
- Do not include numbering, markdown, code fences, or any extra text outside the JSON array.

Strict constraints (must follow all):
- Output is a single JSON containing sentences in [], separated by commas, for example: ["sentence 1", "sentence 2",...]. One string can have many UI test ideas (10-15 ideas) like the example I gave below.
- Absolutely forbidden vague words (example: 'appropriate', 'fast', 'quickly', 'smooth', 'enough', ...). Always give clear, measurable values.
- Steps must be sequential and complete.
- Each sentence must describe multiple attributes of the same element (e.g., font size, color, border, ...).
- Each action must specify multiple attributes at the same time (type, layout, state, ...).
- Always provide specific, measurable expected values with units or counts when applicable: exact pixel sizes (e.g., 16px), RGB/HEX colors (e.g., rgb(34,139,34) or #228B22 or 'green'), timing (e.g., 800ms, 2s, 1 second), percentages (e.g., 75%), URLs (full), screen width (e.g., 360px), coordinates (x,y).
- Do not combine two different actions into the same sentence if they are not related. If the same action applies to two unrelated elements, split them into separate sentences for clarity.
- Inside each description string, use only single quotes (') for character literals; do not use any double quotes (") inside the string content.

Coverage scope (ensure diversity across 20 items):
- You are only allowed to use the following attributes, and no other attributes:
 - Styling: color, background color, font size, font family, font weight, border (px), border radius, opacity, padding, margin
 - State & Interaction: visible/hidden, focused, enabled/disabled, active/selected
 - Position & Layout: x coordinate, y coordinate, width, height, alignment, left, right, top, bottom
 - Multimedia: muted, fullscreen
 - Dynamics: transition, animation, scroll-top/scroll-left
 - Text: text, placeholder, alt text, text align
 - Data: count, value, date
 - Image: source, rendered dimensions, natural dimensions, is loaded, load time, watermark
 - Accessibility: aria-label, label
Each of the 20 description strings must include one or more attributes from the list above. Do not create any attributes outside this list.
Each description MUST only use attributes from the whitelist below:
Attribute whitelist:
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
Output format (must be followed exactly):
- ONLY RETURN a valid JSON array of exactly 20 strings.
- Use JSON double quotes for array items (JSON requirement).
- Inside each string, do not include any double quotes; use only single quotes for quoted labels or literals.
- Do not include numbering, markdown, code fences, or any extra text outside the JSON array.

To help the model understand more generally, please diversify the expression of CSS attributes using synonyms, but still keep the meaning clear and mappable to the correct attribute in the whitelist.
Elements can be in English or Vietnamese, for example: 'Submit' button, 'product_image.jpg' image, 'Happy Holiday' heading. But note that only use English for names, the rest should still be described in English.
Example:
- color can be described as: 'text color', 'font color', 'display color', 'title color', 'icon color'
- background color: 'background', 'behind color', 'background display', 'button background'
- font size: 'text size', 'font magnitude', 'display font size', 'character height'
- font family: 'font type', 'font', 'font set', 'display font type'
- border radius: 'rounded corners', 'corner curve', 'rounded border'
- box shadow: 'shadow', 'block shadow', 'shadow effect', 'blurred shadow behind'
- alignment: 'align', 'center position', 'edge alignment', 'left/right/center align'
- width/height: 'width', 'horizontal extent', 'horizontal size'; 'height', 'display height'
- opacity: 'transparency', 'blur', 'display clarity'
- padding/margin: 'inner spacing', 'padding', 'outer margin', 'surrounding space'

Requirements when using synonyms:
- Must not change the original meaning of the attribute.
- Each test description should alternate different synonyms for the same attribute type to increase language diversity.
- However, in the JSON output, only need to represent as text string (no need to map back).
- The goal is to help the model learn to recognize that many different phrases describe the same UI characteristic.

The description words should be natural like human language, not stereotyped "color = color", but still must be clear and easy to understand.
Valid example: "Verify that the heading has white text color and light blue background, slight rounded corners 6px, center-aligned text."
Do not use single quotes ' or double quotes " randomly. Only use ' when quoting links, text.

Example of 1 string (do not copy):
"Open page 'https://www.shopdemo.com/product/123'. Verify that the product image has source 'hero_image.jpg' and rendered dimensions 400x300px. Ensure the image has alt text containing 'Authentic Shoes'. Verify that exactly 6 thumbnails are displayed below with center alignment and each thumbnail has width 120px and height 80px. Confirm the main heading displays exact text 'Product Title' with font size 18px and color #000000. Verify that the 'Buy Now' button is visible and transitions to selected state with background color #ff0000. Confirm the button has border radius 6px, scroll the page and verify the heading remains fixed (using scroll behavior). Check the language label has aria-label set to English."
Each string needs many test actions like the example above (10-15 actions), with different attributes from the whitelist.
Generate 20 strings, then put them in a JSON array to return. Return in the correct required format (sentences in [], separated by commas, for example: ["sentence 1", "sentence 2",...]).
"""

    all_sentences = []
    for _ in range(5):  # 5 batch để tạo đủ câu
        raw = call_gemini(prompt)
        text = extract_json_from_text(raw) if raw else ""
        sentences = []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                sentences = [str(s) for s in data]
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            pass
        if not sentences:
            sentences = [ln.strip().strip(",") for ln in text.splitlines() if ln.strip()]
        
        for s in sentences:
            s = s.replace('"', "'").strip()
            if not (s.startswith('"') and s.endswith('"')):
                s = f'"{s}"'
            all_sentences.append(s)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",\n".join(all_sentences))
    print(f"✅ Generated {len(all_sentences)} TRAINING sentences -> {out_path}")

def step0_create_test_task(filename: str):
    """Create task using TEST_ATTRIBUTES"""
    INPUT_DIR = Path("Input_Test")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INPUT_DIR / (filename if filename.endswith(".txt") else f"{filename}.txt")
    
    prompt = f"""
You are a professional UI tester.
Generate exactly 20 different UI test descriptions in English.

Output format (must be followed exactly):
- Return ONLY a valid JSON array of exactly 20 strings.
- Use JSON double quotes for the array items (JSON requirement).
- Inside each string, do not include any double quotes; use only single quotes for quoted labels or literals.
- Do not include numbering, markdown, code fences, or any extra text outside the JSON array.

Strict constraints (must follow all):
- Output is a single JSON containing sentences in [], separated by commas, for example: ["sentence 1", "sentence 2",...]. One string can have many UI test ideas (10-15 ideas) like the example I gave below.
- Absolutely forbidden vague words (example: 'appropriate', 'fast', 'quickly', 'smooth', 'enough', ...). Always give clear, measurable values.
- Steps must be sequential and complete.
- Each sentence must describe multiple attributes of the same element (e.g., font size, color, border, ...).
- Each action must specify multiple attributes at the same time (type, layout, state, ...).
- Always provide specific, measurable expected values with units or counts when applicable: exact pixel sizes (e.g., 16px), RGB/HEX colors (e.g., rgb(34,139,34) or #228B22 or 'green'), timing (e.g., 800ms, 2s, 1 second), percentages (e.g., 75%), URLs (full), screen width (e.g., 360px), coordinates (x,y).
- Do not combine two different actions into the same sentence if they are not related. If the same action applies to two unrelated elements, split them into separate sentences for clarity.
- Inside each description string, use only single quotes (') for character literals; do not use any double quotes (") inside the string content.

Coverage scope (ensure diversity across 20 items):
- You are only allowed to use the following attributes, and no other attributes:
 - Styling: color, background color, font size, font family, font weight, border (px), border radius, opacity, padding, margin
 - State & Interaction: visible/hidden, focused, enabled/disabled, active/selected
 - Position & Layout: x coordinate, y coordinate, width, height, alignment, left, right, top, bottom
 - Multimedia: muted, fullscreen
 - Dynamics: transition, animation, scroll-top/scroll-left
 - Text: text, placeholder, alt text, text align
 - Data: count, value, date
 - Image: source, rendered dimensions, natural dimensions, is loaded, load time, watermark
 - Accessibility: aria-label, label
Each of the 20 description strings must include one or more attributes from the list above. Do not create any attributes outside this list.
Each description MUST only use attributes from the whitelist below:
Attribute whitelist:
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
Output format (must be followed exactly):
- ONLY RETURN a valid JSON array of exactly 20 strings.
- Use JSON double quotes for array items (JSON requirement).
- Inside each string, do not include any double quotes; use only single quotes for quoted labels or literals.
- Do not include numbering, markdown, code fences, or any extra text outside the JSON array.

To help the model understand more generally, please diversify the expression of CSS attributes using synonyms, but still keep the meaning clear and mappable to the correct attribute in the whitelist.
Elements can be in English or Vietnamese, for example: 'Submit' button, 'product_image.jpg' image, 'Happy Holiday' heading. But note that only use English for names, the rest should still be described in English.
Example:
- color can be described as: 'text color', 'font color', 'display color', 'title color', 'icon color'
- background color: 'background', 'behind color', 'background display', 'button background'
- font size: 'text size', 'font magnitude', 'display font size', 'character height'
- font family: 'font type', 'font', 'font set', 'display font type'
- border radius: 'rounded corners', 'corner curve', 'rounded border'
- box shadow: 'shadow', 'block shadow', 'shadow effect', 'blurred shadow behind'
- alignment: 'align', 'center position', 'edge alignment', 'left/right/center align'
- width/height: 'width', 'horizontal extent', 'horizontal size'; 'height', 'display height'
- opacity: 'transparency', 'blur', 'display clarity'
- padding/margin: 'inner spacing', 'padding', 'outer margin', 'surrounding space'

Requirements when using synonyms:
- Must not change the original meaning of the attribute.
- Each test description should alternate different synonyms for the same attribute type to increase language diversity.
- However, in the JSON output, only need to represent as text string (no need to map back).
- The goal is to help the model learn to recognize that many different phrases describe the same UI characteristic.

The description words should be natural like human language, not stereotyped "color = color", but still must be clear and easy to understand.
Valid example: "Verify that the heading has white text color and light blue background, slight rounded corners 6px, center-aligned text."
Do not use single quotes ' or double quotes " randomly. Only use ' when quoting links, text.

Example of 1 string (do not copy):
"Open page 'https://www.shopdemo.com/product/123'. Verify that the product image has source 'hero_image.jpg' and rendered dimensions 400x300px. Ensure the image has alt text containing 'Authentic Shoes'. Verify that exactly 6 thumbnails are displayed below with center alignment and each thumbnail has width 120px and height 80px. Confirm the main heading displays exact text 'Product Title' with font size 18px and color #000000. Verify that the 'Buy Now' button is visible and transitions to selected state with background color #ff0000. Confirm the button has border radius 6px, scroll the page and verify the heading remains fixed (using scroll behavior). Check the language label has aria-label set to English."
Each string needs many test actions like the example above (10-15 actions), with different attributes from the whitelist.
Generate 20 strings, then put them in a JSON array to return. Return in the correct required format (sentences in [], separated by commas, for example: ["sentence 1", "sentence 2",...]).
"""

    all_sentences = []
    for _ in range(5):  # 5 batch để tạo đủ câu
        raw = call_gemini(prompt)
        text = extract_json_from_text(raw) if raw else ""
        sentences = []
        try:
            data = json.loads(text)
            if isinstance(data, list):
                sentences = [str(s) for s in data]
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            pass
        if not sentences:
            sentences = [ln.strip().strip(",") for ln in text.splitlines() if ln.strip()]
        
        for s in sentences:
            s = s.replace('"', "'").strip()
            if not (s.startswith('"') and s.endswith('"')):
                s = f'"{s}"'
            all_sentences.append(s)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(",\n".join(all_sentences))
    print(f"✅ Generated {len(all_sentences)} TEST sentences -> {out_path}")

def step1_analyze_task(input_file, batch_size=20):
    full_text = load_text(input_file)
    main_tasks = re.findall(r'"(.*?)"', full_text, re.DOTALL)
    print(len(main_tasks))
    task_list = []
    task_trace = []

    for start in range(0, len(main_tasks), batch_size):
        batch = main_tasks[start:start + batch_size]
        prompt = f"""You are a professional UI test case generator.
Your job is to convert the following natural language test description into a list of clear, atomic UI actions.
Follow these strict rules:
- Output must be a JSON array of arrays.
- Group related properties of the **same UI element in the same state** into one atomic task.
  - This includes all visual, layout, formatting, style, alignment, and content properties (e.g. font size, color, background, border radius, alignment).
  - Do not split alt text, watermark, natural vs rendered size, sharpness/compression, etc — if all refer to the same image in its normal state.
  - Do not separate things like color, format, alignment, font, padding, size, position, compair with another element if they belong to the same element in the same context.
  - Only separate tasks if:
    - The properties relate to **different states** (e.g., hover vs normal).
    - The properties relate to **different elements**.
    - The instruction involves a **user action** followed by a **verification** (e.g., click then check, locate then check, locate then type), or an explicit user action in between is required.
    - The instruction involves many actions (example: "Locate the Email input field and enter 'invalid-email'" must be separated into 2 tasks: Locate the Email input field, Enter 'invalid-email' into the Email input field).
    - The behavior differs between **device types** (e.g., mobile vs desktop).
    - Action requires many steps to complete (example: Log in with the account 'patient001' and password 'Health@2023' must be separated into 2 tasks: Enter 'patient001' into the account field, Enter 'Health@2023' into the password field).
  - (Example: The 'Add to Cart' button has a color of '#ff9900', a border radius of '4px', and the background color changes to '#e68a00' on hover)
- Each array item is a single atomic test action string.
- Do not generate steps that are not described in the main task.
- Do NOT include explanations, markdown, comments, or anything else — only the JSON array.

Example:
- Requirement:
"Open 'https://www.netflix.com', click the Sign In button located at the top-right corner (50px from the top edge and 30px from the right edge), enter the email 'testuser@example.com' and password 'Test@1234', submit the login form, verify that after submission, the user is redirected to 'https://www.netflix.com/browse' within 3 seconds and the page displays at least 5 personalized movie thumbnails, and ensure that all input fields have the font family Roboto with size 16px, padding 12px, bottom margin 16px; the Sign In button has a background color '#e50914', text color '#ffffff', border radius 4px, is responsive down to a 360px screen width, and all elements meet a WCAG contrast ratio of at least 4.5:1.",
"Open 'https://www.amazon.com', search for 'Dàn loa Bluetooth', verify that the search completes in under 2 seconds, displays at least 10 products with titles containing 'Loa Bluetooth', confirm each product card has an image width of exactly '150px', product price has a font size of '18px', and ensure the 'Add to Cart' button has a color '#ff9900', rounded corners '4px', and hover background color changes to '#e68a00'.",
"Open 'https://shop.example.com/item/P123', verify that the main product image '#main-product-image' loads successfully within 800ms, has alt text containing 'Ví da Premium', no watermark, natural dimensions 1600x1200, displayed at 400x300 with a tolerance of ±4px, sharpness ≥ 0.85, compression artifacts ≤ 0.1, and ensure the image is positioned to the left of the price block '.price' with a minimum spacing of 24px (±2px)."
- Response:
[
  "Open 'https://www.netflix.com'",
  "Locate the Sign In button at the top-right corner (50px from the top edge and 30px from the right edge)",
  "Click the Sign In button",
  "Enter 'testuser@example.com' into the email input field",
  "Enter 'Test@1234' into the password input field",
  "Submit the login form",
  "Verify that after submission, the user is redirected to 'https://www.netflix.com/browse' within 3 seconds",
  "Verify that the page displays at least 5 personalized movie thumbnails",
  "Verify that all input fields have the font 'Roboto 16px', padding '12px', bottom margin '16px'",
  "Verify that the Sign In button has a background color '#e50914', text color '#ffffff', border radius 4px",
  "Verify that the Sign In button is responsive down to a 360px screen width",
  "Verify that all elements meet a WCAG contrast ratio of at least 4.5:1"
],
[
  "Open 'https://www.amazon.com'",
  "Search for 'Dàn loa Bluetooth'",
  "Verify that the search completes in under 2 seconds",
  "Verify that the search results display at least 10 products with titles containing 'Loa Bluetooth'",
  "Verify that each product card has an image width of exactly '150px'",
  "Verify that the product price has a font size of '18px'",
  "Verify that the Add to Cart button has a color '#ff9900' and rounded corners 4px",
  "Verify that the Add to Cart button's background color changes to '#e68a00' on hover"
],
[
  "Open 'https://shop.example.com/item/P123'",
  "Verify that the main product image '#main-product-image' loads successfully within 800ms, has alt text containing 'Ví da Premium', no watermark, natural dimensions 1600x1200, displayed at 400x300 with a tolerance of ±4px, sharpness ≥ 0.85, compression artifacts ≤ 0.1, and ensure the image is positioned to the left of the price block '.price' with a minimum spacing of 24px (±2px)"
]

Now generate the JSON from this requirement:
- Requirement:
{chr(10).join([f'{i+1}. "{t}"' for i, t in enumerate(batch)])}

Now return only the JSON array of arrays, in the same order.
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
                # If still not a list-of-lists with matching length → error
                if not isinstance(result, list) or len(result) != len(batch) or any(not isinstance(x, list) for x in result):
                    raise ValueError("Model did not return a valid array of arrays with matching length.")
                # Thêm từng requirement và list subtasks tương ứng
                for req_text, subtasks in zip(batch, result):
                    task_list.append(subtasks)
                    task_trace.append((req_text, subtasks))
                break
            except Exception as e:
                print(f"❌ Error (attempt {attempt + 1}/3): {e}")
                print(f"⚠️ Raw response: {response}")
                if attempt == retries - 1:
                    # Fallback: vẫn giữ chỗ bằng mảng rỗng cho đúng số lượng
                    for req_text in batch:
                        task_list.append([])
                        task_trace.append((req_text, []))
                    print("⚠️ Continued with empty results for this batch.")

    return task_list, task_trace

def step2_generate_steps(subtask_groups):
    all_step_groups = []
    step_trace = []
    for subtask_list in subtask_groups:    
        prompt = f"""You are a professional UI test step generator.
Your task is to convert the following atomic test instruction into a list of executable UI test steps in JSON format (must in English).

Instruction: "{subtask_list}"

Each step must follow this JSON structure:
{{
  "action": "action to perform (e.g. click, type, hover, locate, verify, etc.)",
  "selector": "CSS selector or a natural description (keep Vietnamese text/labels if present in the instruction, e.g. 'Login' button), or empty string if not applicable",
  "value": "string value to type or expect, or empty string if not applicable, keep Vietnamese text if provided",
  "expected": {{
    // This object may contain multiple properties,
    // based only on what the instruction requires.
    // Each key is the name of a UI property (e.g. color, font, position).
    // Each value is the expected value or structured object.
  }}
}}

Explanation of the actions:
- click, type, hover, select, submit forms, search, goto etc. (must always return 'status')
- locate: find, identifying UI elements based on text, position (e.g. near/above), or attributes (class, ID, etc.).
- verify: validating properties like color, alignment, text, count, image presence, OCR output, size, position, etc. Checking that something exists, is visible, absent, or meets a layout/appearance requirement.

Important:
- Always return output in JSON with English keys.
- All fields (`action`, `selector`, `value`, `expected`) **must always be present**.
- If a field has no value, use:
  - `""` for empty strings
  - `{{}}` for empty object (when `expected` is not needed)
- Do not omit any field from the JSON step.
- Do not include any property in `expected` if it is not clearly mentioned in the instruction.
- `expected` must be a valid object (`{{}}`), and can contain:
  - 'status': boolean (true if the action succeeds, false if it fails)
  - 'text':
  - 'placeholder':
  - 'language': e.g., `"French"`, `"Chinese"`
  - 'position': object with `x`, `y`, `width`, `height`, etc.. (e.g., 'position: {{"x": 100, "y": 200, "width": "300px", "height": "150px"}}')
  - 'overflow': boolean (true if content overflows, false if not)
  - 'occluded': boolean (true if element is occluded by another, false if not)
  - 'styles':
    - 'textAlign':
    - `color`: e.g., "#ffffff" or "rgb(255, 255, 255)" or "white"
    - `background color`
    - `font`: an object with `family`, `size`, `weight`, `style` if mentioned 
    - 'border': width, style, color, radius
    - 'padding': top, right, bottom, left
    - 'margin': top, right, bottom, left
    - 'alignment': "left", "center", "right", "justify", or horizontal/vertical alignment in pixels
    - gapPx: e.g., "10px"
    - 'opacity': "0.5", "1", etc.
    - Or any property related to 'styles', as long as it follows the correct format and rules stated (only declare if present in the input description).
  - 'state':
    - `visibility`: "visible" or "hidden"
    - `enabled`: boolean (true or false)
    - 'focused': boolean
    - 'hovered': boolean     
    - 'active': boolean
    - Or any property related to 'state', as long as it follows the correct format and rules stated (only declare if present in the input description).   
  - dynamics:
    - `movement`: "static" or "moves" after an action
    - `scroll`: "sticky" | "fixed" | "static" | "none" or something
    - 'parallax': number (e.g., `0.5` for parallax effect)
    - `transition`: boolean (CSS transition)
    - 'animate': boolean (motion effect)
    - `focus`: boolean, ('true' if element should be focused, via Tab or click)
    - `navigation`: destination URL if checking page redirection
    - `loading`: `"present", "loading", "none"` for spinners, indicators
    - 'progressValue': string, (% value if it is progress bar, pie chart, etc)
    - Or any property related to 'dynamics', as long as it follows the correct format and rules stated (only declare if present in the input description).
  - If the element is image (img), its may contain:
    - `source`: URL of the image
    - 'alt text': string (e.g., "Hình ảnh sản phẩm", "Logo")
    - 'is loaded': boolean (true if image is loaded, false if not)
    - 'load time': number (in milliseconds, e.g., 500 for 0.5 seconds)
    - 'sharpness score GTE': number (0 to 1, e.g., 0.8 for sharpness)
    - 'compression artifacts LTE': number (0 to 1, e.g., 0.2 for compression artifacts)
    - 'rendered dimensions': object with `width`, `height`, 'tolerance Px' (in pixels, e.g., 300x200)
    - 'natural dimensions': object with `width`, `height` (natural size of the image)
    - 'viewport dimensions': object with `width`, `height` (size of the image in the viewport)
    - 'watermark': boolean (true if image has a watermark, false if not)
  - 'compair': if comparing with another element, it may contain:
    - 'element': object describing the element to compare with (must always have when `compare` is used). (if there are multiple elements to compare, use 'element1', 'element2', etc.)
      - This object may contain `selector` (required), and may also include `action`, `value`, 'expected' only if explicitly described in the task.  
      - `expected` follows the same structure and rules as the `expected` field for the main element.
    - Properties to compare (e.g., `color`, `font size`, `position`, etc.)
  - Or any property related to expected , as long as it follows the correct format and rules stated (only declare if present in the input description).

Example 1:
- Task list:
[
  "Open 'https://www.amazon.com'",
  "Search for 'Dàn loa Bluetooth'",
  "Verify that the search completes in under 2 seconds",
  "Verify that the search results display at least 10 products with titles containing the word 'Loa Bluetooth'",
  "Verify that each product card has a main image width exactly '150px'",
  "Verify that the 'Add to Cart' button has color '#ff9900' and border radius '4px', positioned at x=120px and y=300px, with width '200px' and height '50px'",
  "Verify that the 'Add to Cart' button background color changes to '#e68a00' when hovered"
]
- Response:
[
  {{
    "action": "goto",
    "selector": "",
    "value": "https://www.amazon.com",
    "expected": {{
      "status": true,
      "navigation": "https://www.amazon.com"
    }}
  }},
  {{
    "action": "search",
    "selector": "'Search' input field",
    "value": "Dàn loa Bluetooth",
    "expected": {{
      "status": true
    }}
  }},
  {{
    "action": "verify",
    "selector": "search results",
    "value": "",
    "expected": {{
      "state": {{
        "visibility": "visible"
      }},
      "dynamics": {{
        "duration": 2000
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": ".search-results .product-title",
    "value": "",
    "expected": {{
      "count": {{
        "min": 10
      }},
      "text": {{
        "contains": "Loa Bluetooth"
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "'product card'",
    "value": "",
    "expected": {{
      "img": {{
        "styles": {{
          "width": "150px"
        }}
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Add to Cart' button",
    "value": "",
    "expected": {{
      "text": "Add to Cart",
      "styles": {{
        "color": "#ff9900",
        "border": {{
          "radius": "4px"
        }}
      }},
      "position": {{
        "x": "120px",
        "y": "300px",
        "width": "200px",
        "height": "50px"
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Add to Cart' button",
    "value": "",
    "expected": {{
      "text": "Add to Cart",
      "state": {{
        "hovered": true
      }},
      "styles": {{
        "background color": "#e68a00"
      }}
    }}
  }}
]

Example 2:
- Task list:
[
  "Locate the 'Login' button at the top right corner (50px from the top and 30px from the right)",
  "Click the 'Login' button",
  "Type 'testuser@example.com' into the email input field",
  "Type 'Test@1234' into the password input field",
  "Submit the login form",
  "Verify that after submission, the user is redirected to 'https://www.netflix.com/browse' within 3 seconds",
  "Verify that the page displays at least 5 personalized movie thumbnails",
  "Verify that all input fields have font 'Roboto 16px', padding '12px', and margin-bottom '16px'",
  "Verify that the 'Sign In' button has background color '#e50914', text color '#ffffff', and border radius '4px'",
  "Verify that the 'Sign In' button is responsive down to screen width 360px",
  "Verify that all visible elements meet a WCAG contrast ratio of at least 4.5:1"
]
- Response:
[
  {{
    "action": "locate",
    "selector": "'Login' button",
    "value": "",
    "expected": {{
      "status": true,
      "text": "Login",
      "position": {{
        "top": "50px",
        "right": "30px"
      }}
    }}
  }},
  {{
    "action": "click",
    "selector": "'Login' button",
    "value": "",
    "expected": {{
      "status": true,
    }}
  }}
  {{
    "action": "type",
    "selector": "email input field",
    "value": "testuser@example.com",
    "expected": {{
      "status": true
    }}
  }},
  {{
    "action": "type",
    "selector": "password input field",
    "value": "Test@1234",
    "expected": {{
      "status": true
    }}
  }},
  {{
    "action": "submit",
    "selector": "login form",
    "value": "",
    "expected": {{
      "status": true,
      "dynamics": {{
        "submit": true
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "",
    "value": "",
    "expected": {{
      "dynamics": {{
        "navigation": "https://www.netflix.com/browse",
        "timeout": 3000
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "personalized movie thumbnails",
    "value": "",
    "expected": {{
      "count": {{
        "min": 5
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "input",
    "value": "",
    "expected": {{
      "styles": {{
        "font": {{
          "family": "Roboto",
          "size": "16px"
        }},
        "padding": {{
          "top": "12px",
          "right": "12px",
          "bottom": "12px",
          "left": "12px"
        }},
        "margin": {{
          "bottom": "16px"
        }}
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Login' button",
    "value": "",
    "expected": {{
      "text": "Login",
      "styles": {{
        "background color": "#e50914",
        "color": "#ffffff",
        "border": {{
          "radius": "4px"
        }}
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Login' button",
    "value": "",
    "expected": {{
      "text": "Login",
      "screenWidth": "360px",
      "state": {{
        "visibility": "visible",
        "enabled": true
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "all visible elements",
    "value": "",
    "expected": {{
      "accessibility": {{
        "contrastRatio": "at least 4.5:1"
      }}
    }}
  }}
]

Example 3:
- Task list:
[
  "Open 'https://shop.example.com/item/P123'",
  "Verify that the main product image '#main-product-image' loads successfully within 800ms, has alt text containing 'Premium Leather Wallet', has no watermark, natural size 1600x1200, renders at 400x300 with tolerance ±4px, sharpness ≥ 0.85 and compression artifacts ≤ 0.1, and ensure the image is positioned to the left of the price block '.price' with a minimum gap of 24px (±2px)"
]
- Response:
[
  {{
    "action": "goto",
    "selector": "",
    "value": "https://shop.example.com/item/P123",
    "expected": {{
      "status": true,
      "navigation": "https://shop.example.com/item/P123"
    }}
  }},
  {{
    "action": "verify",
    "selector": "#main-product-image",
    "value": "",
    "expected": {{
      "isLoaded": true,
      "loadTime": {{
        "max": 800
      }},
      "alt text": {{
        "contains": "Premium Leather Wallet"
      }},
      "watermark": false,
      "natural dimensions": {{
        "width": 1600,
        "height": 1200
      }},
      "rendered dimensions": {{
        "width": 400,
        "height": 300,
        "tolerance Px": 4
      }},
      "sharpness score GTE": {{
        "min": 0.85
      }},
      "compression artifacts LTE": {{
        "max": 0.1
      }},
      "compair": {{
        "element": {{
          "selector": ".price"
        }},
        "position": "left",
        "gapPx": {{
          "min": 24,
          "tolerancePx": 2
        }}
      }}
    }}
  }}
]

Example 4:
- Task list:
[
  "Click on contact 'Nguyen Van An'",
  "Type 'Hello' into the message input field with height '40px' and font size '14px'",
  "Send the message",
  "Verify that the sent message appears in the chat window within 1 second, with background color '#dcf8c6', right-aligned, and the message timestamp formatted in '12-hour' style, aligned to the bottom-right corner inside the chat window"
]

- Response:
[
  {{
    "action": "click",
    "selector": "'Nguyen Van An' contact",
    "value": "",
    "expected": {{
      "status": true
    }}
  }},
  {{
    "action": "type",
    "selector": "'message' input",
    "value": "Hello",
    "expected": {{
      "status": true,
      "styles": {{
        "height": "40px",
        "font size": "14px"
      }},
    }}
  }},
  {{
    "action": "submit",
    "selector": "'send message' button",
    "value": "",
    "expected": {{
      "status": true
    }}
  }},
  {{
    "action": "verify",
    "selector": "'sent message bubble'",
    "value": "",
    "expected": {{
      "state": {{
        "visibility": true,
        "alignment": {{
          'horizontal': 'right',
          'vertical': 'bottom'
        }}
      }},
      'styles': {{
        'background color': '#dcf8c6',
      }},
      'dynamics': {{
        'timeout': 1000
      }},
      'timestamp': {{
        'format': '12-hour',
        'alignment': 'bottom-right'
      }}
    }}
  }}
]

Example 5:
- Task list:
[
  "Verify that the error message text 'Invalid account.' is displayed",
  "Verify that the font size and color of the error message are exactly the same as the font size and color of the 'Account' label and the 'Password' label, and ensure that both are exactly 20px and '#ff0000'"
]
- Response:
[
  {{
    "action": "verify",
    "selector": "error message",
    "value": "",
    "expected": {{
      "text": "Invalid account.",
      "state": {{
        "visibility": "visible"
      }}
    }}
  }},
  {{
    "action": "verify",
    "selector": "error message",
    "value": "",
    "expected": {{
      "style": {{
        "font": {{
          "size": "20px"
        }},
        "color": "#ff0000"
      }},
      "compair": {{
        "element1": {{
          "selector": "'Account' label",
          "expected": {{
            "style": {{
              "font": {{
                "size": "20px"
              }},
              "color": "#ff0000"
            }}
          }}
        }},
        "element2": {{
          "selector": "'Password' label",
          "expected": {{
            "style": {{
              "font": {{
                "size": "20px"
              }},
              "color": "#ff0000"
            }}
          }}
        }}
      }}
    }}
  }}
]

Notes:
- Each atomic test instruction return only ONE test steps, NO splitting the instruction into multiple substeps.
- The selector can be a CSS selector or a natural-language reference to the element.
- UI may be implemented using any frontend framework (e.g., TailwindCSS, Bootstrap, raw HTML).
- Use single quotes ONLY for literal string values (URLs, hex colors, text content, file names). Do NOT quote CSS property values like font names, alignment values, UI element names, or attribute names.
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
                # Kiểm tra cơ bản: mảng và cùng độ dài với subtask_list
                if not isinstance(steps, list) or len(steps) != len(subtask_list):
                    raise ValueError("Model did not return a JSON array with the same length as input.")
                # Bảo đảm mỗi phần tử có đủ 4 khóa, điền mặc định nếu thiếu
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

                    # Lưu trace (mỗi subtask -> đúng 1 step)
                    step_trace.append((subtask_list[idx], [step_obj]))
                step_group = normalized
                break
            except Exception as e:
                print(f"❌ Error (attempt {attempt + 1}/3): {e}")
                print(f"⚠️ Raw response: {response}")
                if attempt == retries - 1:
                    # Fallback: tạo step rỗng giữ chỗ (để pipeline không gãy)
                    step_group = [{
                        "action": "",
                        "selector": "",
                        "value": "",
                        "expected": {}
                    } for _ in subtask_list]
        all_step_groups.append(step_group)
    return all_step_groups, step_trace

def save_excel_summary(filename, task_trace, step_groups, file_type):
    rows = []
    for idx, (main_task, subtasks) in enumerate(task_trace):
        step_group = step_groups[idx] if idx < len(step_groups) else []
        subtask_json = json.dumps(subtasks, ensure_ascii=False)
        step_json = json.dumps(step_group, ensure_ascii=False)
        rows.append({
            "File_Type": file_type,
            "Main Task": main_task,
            "Sub Tasks": subtask_json,
            "Steps": step_json
        })
    df = pd.DataFrame(rows)
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(filename, index=False)

def main():
    #global KEY_LIST
    inputName = input("Enter the base name for input files: ")
    #key1 = input("Enter the Gemini API key1: ")
    #key2 = input("Enter the Gemini API key2: ")
    #key3 = input("Enter the Gemini API key3: ")
    #key4 = input("Enter the Gemini API key4: ")
    #key5 = input("Enter the Gemini API key5: ")
    #key6 = input("Enter the Gemini API key6: ")
    #key7 = input("Enter the Gemini API key7: ")
    #key8 = input("Enter the Gemini API key8: ")
    #KEY_LIST = [key1, key2, key3, key4, key5, key6, key7, key8]

    # Tạo thư mục
    TASK_DIR_TRAIN = Path("JSONtask_Train")
    STEP_DIR_TRAIN = Path("JSONwStep_Train")
    REPORT_DIR_TRAIN = Path("Report_Train")
    
    TASK_DIR_TEST = Path("JSONtask_Test")
    STEP_DIR_TEST = Path("JSONwStep_Test")
    REPORT_DIR_TEST = Path("Report_Test")
    
    for d in [TASK_DIR_TRAIN, STEP_DIR_TRAIN, REPORT_DIR_TRAIN, 
              TASK_DIR_TEST, STEP_DIR_TEST, REPORT_DIR_TEST]:
        d.mkdir(parents=True, exist_ok=True)

    numFiles = 20
    print(f"Generate {numFiles} files with 3:1 train:test ratio")
    
    for i in range(numFiles):
        # Tạo file theo tỉ lệ 3:1 (3 train, 1 test)
        if (i + 1) % 4 == 0:  # File thứ 4, 8, 12, 16, 20 là test
            file_type = "test"
            filename = f"{inputName}_test_{i}.txt"
            INPUT_DIR = Path("Input_Test")
            
            if not (INPUT_DIR / filename).exists():
                print(f"Creating TEST file: {filename}")
                step0_create_test_task(filename)
            else:
                print(f"TEST file already exists: {filename}")
                
            # Process test file
            case_id = Path(filename).stem
            try:
                print(f"\n▶️ Running TEST pipeline for: {case_id}")
                task_file = INPUT_DIR / f"{case_id}.txt"
                task_json_file = TASK_DIR_TEST / f"{case_id}.task.json"
                task_list, task_trace = step1_analyze_task(task_file)
                save_json(task_json_file, task_list)
                print("\n Step 1 TEST success \n")

                step_json_file = STEP_DIR_TEST / f"{case_id}.step.json"
                step_groups, step_trace = step2_generate_steps(task_list)
                save_json(step_json_file, step_groups)
                print("\n Step 2 TEST success \n")

                report_file = REPORT_DIR_TEST / f"{case_id}.summary.xlsx"
                save_excel_summary(report_file, task_trace, step_groups, "TEST")
                print(f"✅ TEST Excel saved to: {report_file}")
            except Exception as e:
                print(f"❌ Error in TEST {case_id}: {e}")
                
        else:  # File train
            file_type = "train"
            filename = f"{inputName}_train_{i}.txt"
            INPUT_DIR = Path("Input_Train")
            
            if not (INPUT_DIR / filename).exists():
                print(f"Creating TRAIN file: {filename}")
                step0_create_train_task(filename)
            else:
                print(f"TRAIN file already exists: {filename}")
                
            # Process train file
            case_id = Path(filename).stem
            try:
                print(f"\n▶️ Running TRAIN pipeline for: {case_id}")
                task_file = INPUT_DIR / f"{case_id}.txt"
                task_json_file = TASK_DIR_TRAIN / f"{case_id}.task.json"
                task_list, task_trace = step1_analyze_task(task_file)
                save_json(task_json_file, task_list)
                print("\n Step 1 TRAIN success \n")

                step_json_file = STEP_DIR_TRAIN / f"{case_id}.step.json"
                step_groups, step_trace = step2_generate_steps(task_list)
                save_json(step_json_file, step_groups)
                print("\n Step 2 TRAIN success \n")

                report_file = REPORT_DIR_TRAIN / f"{case_id}.summary.xlsx"
                save_excel_summary(report_file, task_trace, step_groups, "TRAIN")
                print(f"✅ TRAIN Excel saved to: {report_file}")
            except Exception as e:
                print(f"❌ Error in TRAIN {case_id}: {e}")

    print("\n" + "="*50)
    print("SUMMARY:")
    print(f"✅ Generated {numFiles} files total")
    print(f"📚 TRAIN files: {len([i for i in range(numFiles) if (i+1) % 4 != 0])}")
    print(f"🧪 TEST files: {len([i for i in range(numFiles) if (i+1) % 4 == 0])}")
    print(f"📊 TRAIN attributes: {len(TRAIN_ATTRIBUTES)}")
    print(f"📊 TEST attributes: {len(TEST_ATTRIBUTES)}")
    print("🔒 Zero leakage between train/test attribute sets")

if __name__ == "__main__":

    main()
