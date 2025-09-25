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
    "AIzaSyBxq1XVTElXyWYhqBunCAXBaGXAhHCr9hg"
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
You are a professional UI tester specialized in creating TRAINING data.
Generate exactly 20 different UI test descriptions in English for TRAINING SET.

STRICT CONSTRAINTS FOR TRAINING SET:
===================================
- Each description is a single JSON string item, but may contain multiple sentences (10-15 sentences).
- ABSOLUTELY FORBID vague words (example: 'correctly', 'appropriately', 'fast', 'quickly', 'smooth', 'enough', ...). 
- ALWAYS give explicit, measurable values.
- STRICTLY FORBIDDEN to make up or use any attributes outside the allowed list.
- DO NOT use any attributes other than the list below.
- Use single quotes ONLY for literal string values (URLs, hex colors, text content, file names). Do NOT quote CSS property values like font names, alignment values, UI element names, or attribute names.

ALLOWED ATTRIBUTES FOR TRAINING SET (ONLY USE THESE ATTRIBUTES):
==============================================================
Styles: 'color', 'background color', 'font family', 'font size', 'font weight', 'text alignment', 'opacity', 'border', 'padding', 'margin', 'gap'
States & Interactivity: 'visible/hidden', 'enabled/disabled', 'focused', 'hovered', 'active/selected'
Position & Layout: 'x/y coordinates', 'width', 'height', 'alignment (absolute/relative)', 'gap', 'aspect ratio', 'gap Px', 'offset', 'tolerance Px', 'z index', 'occluded', 'overflow'
Content: 'text', 'placeholder', 'language'
Media: 'muted', 'playing', 'fullscreen'
Dynamics: 'transition', 'animation', 'movement', 'parallax', 'progress direction', 'scroll', 'loading', 'progress value'
Text: 'text validation', 'placeholder text', 'allowed characters', 'text content'
Data: 'count', 'value', 'time value'
Image: 'source', 'is loaded', 'load time', 'sharpness score GTE', 'compression artifacts LTE', 'watermark', 'natural dimensions'
Accessibility: 'activatable by keyboard', 'keyboard focusable', 'keyboard navigable', 'focus trapping', 'label'
Repetitive & Hierarchical: 'list/table validation'
Complex Visual: 'shape validation', 'image composition'
More: 'flowchart validation', 'browser compatibility'

ABSOLUTELY MUST NOT use the following attributes (reserved for test set):
- fill, visited state color, line height, border radius, box shadow
- clickable, cursor, transform, clipping, blurring, left/right, top/bottom
- exists, present, source type, duration, auto advance, auto play, timeout
- language detection, alt text, date, currency symbol
- rendered dimensions, viewport dimensions, video/audio, aria-label, contrast ratio
- nested elements check, graphics/charts validation, multiple attribute comparison

STRICT REQUIREMENTS:
==================
- The steps should be sequential and complete.
- Each sentence must describe multiple attributes of the same element OR specify at least one explicit spatial constraint.
- Always provide concrete values: exact pixels (16px), RGB/HEX colors (#228B22), timing (800ms), percentages (75%), full URLs, coordinates (x,y), gaps/tolerance (24px ±2px).
- Inside each description string, use single quotes only for literal string values like URLs, hex colors, and text content. Do NOT quote CSS property values, font names, alignment values, UI element names, or attribute names.
- Correct: "verify the Submit button has background color '#ff0000', font family Roboto, and text alignment left" 
- Incorrect: "verify the 'Submit' button has 'background color' '#ff0000', 'font family' 'Roboto', and 'text alignment' 'left'"

OUTPUT FORMAT:
=============
- Return ONLY a valid JSON array of exactly 20 strings.
- Use JSON double quotes for array items.
- Do not include numbering, markdown, code fences, or extra text.

TRAINING EXAMPLE (for reference only):
"Open 'https://shop.demo.com/product/456', verify that the product image has source equal to 'main_product.jpg' and natural dimensions of 350x250px ±3px, check the product title shows exact text 'Premium Quality Product' with font size 20px and color '#333333', confirm the Buy Now button has width 180px, height 45px, border radius 8px, z index of 15, and changes to hovered state with background color '#ff6600', ensure transition animation duration of 250ms applies, scroll the page and verify header remains fixed with absolute alignment, check that exactly count 8 related products are displayed."

IMPORTANT NOTES: 
1. Always use SEPARATE WORDS for compound attributes. Use "background color" instead of "backgroundColor", "font size" instead of "fontSize", "natural dimensions" instead of "naturalDimensions", etc.
2. Use single quotes only for literal string values like URLs, hex colors, and text content. Do NOT quote CSS property values like font names (Roboto), alignment values (left, center), UI element names, or attribute names.
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
You are a professional UI tester specialized in creating TESTING data.
Generate exactly 20 different UI test descriptions in English for TEST SET.

STRICT CONSTRAINTS FOR TEST SET:
================================
- Each description is a single JSON string item, but may contain multiple sentences (10-15 sentences).
- ABSOLUTELY FORBID vague words (example: 'correctly', 'appropriately', 'fast', 'quickly', 'smooth', 'enough', ...).
- ALWAYS give explicit, measurable values.
- STRICTLY FORBIDDEN to make up or use any attributes outside the allowed list.
- DO NOT use any attributes other than the list below.
- Use single quotes ONLY for literal string values (URLs, hex colors, text content, file names). Do NOT quote CSS property values like font names, alignment values, UI element names, or attribute names.

DESIGNATED ATTRIBUTES FOR TEST SET (ONLY USE THESE ATTRIBUTES):
=============================================================
Styles: 'fill', 'visited state color', 'line height', 'border radius', 'box shadow'
States & Interactivity: 'clickable', 'cursor'
Position & Layout: 'transform', 'clipping', 'blurring', 'left/right', 'top/bottom'
Content: 'exists', 'present'
Media: 'source type'
Dynamics: 'duration', 'auto advance', 'auto play', 'timeout'
Text: 'language detection', 'alt text'
Data: 'date', 'currency symbol'
Image: 'rendered dimensions', 'viewport dimensions', 'video/audio'
Accessibility: 'aria-label', 'contrast ratio'
Repetitive & Hierarchical: 'nested elements check'
Complex Visual: 'graphics/charts validation'
More: 'multiple attribute comparison'

ABSOLUTELY MUST NOT use the following training attributes:
- color, background color, font family, font size, font weight, text alignment, opacity, border, padding, margin, gap
- visible/hidden, enabled/disabled, focused, hovered, active/selected
- x/y coordinates, width, height, alignment (absolute/relative), aspect ratio, gap Px, offset, tolerance Px, z index, occluded, overflow
- text, placeholder, language, muted, playing, fullscreen
- transition, animation, movement, parallax, progress direction, scroll, loading, progress value
- text validation, placeholder text, allowed characters, text content, count, value, time value
- source, is loaded, load time, sharpness score GTE, compression artifacts LTE, watermark, natural dimensions
- activatable by keyboard, keyboard focusable, keyboard navigable, focus trapping, label
- list/table validation, shape validation, image composition, flowchart validation, browser compatibility

STRICT REQUIREMENTS:
==================
- The steps should be sequential and complete.
- Each sentence must describe multiple attributes of the same element OR specify at least one explicit spatial constraint.
- Always provide concrete values: exact ratios (4.5:1), specific aria-label, play states (playing/paused), specific shapes (circle/square), exact altText.
- Inside each description string, use single quotes only for literal string values like URLs, hex colors, and text content. Do NOT quote CSS property values, font names, alignment values, UI element names, or attribute names.
- Correct: "check navigation menu aria-label is set to 'Main Navigation Menu' and has font family Arial with text alignment center"
- Incorrect: "check 'navigation menu' 'aria-label' is set to 'Main Navigation Menu' and has 'font family' 'Arial' with 'text alignment' 'center'"

OUTPUT FORMAT:
=============
- Return ONLY a valid JSON array of exactly 20 strings.
- Use JSON double quotes for array items.
- Do not include numbering, markdown, code fences, or extra text.

TEST EXAMPLE (for reference only):
"Open 'https://media.example.com/player/789', verify that the video player is in playing state, check image has alt text containing 'New Product Introduction Video 2024', confirm nested elements are checked in proper order from parent to child with max depth 3 levels, ensure all text elements have contrast ratio at least 4.5:1 against background, check navigation menu aria-label is set to 'Main Navigation Menu', verify button has shape of circle with exact radius, compare multiple attribute between header and footer for accessibility compliance."

IMPORTANT NOTES:
1. Always use SEPARATE WORDS for compound attributes. Use "alt text" instead of "altText", "contrast ratio" instead of "contrastRatio", "rendered dimensions" instead of "renderedDimensions", etc.
2. Use single quotes only for literal string values like URLs, hex colors, and text content. Do NOT quote CSS property values like font names (Arial), alignment values (center, left), UI element names, or attribute names.
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