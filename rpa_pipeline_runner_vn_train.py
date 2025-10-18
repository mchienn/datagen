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

def step0_create_task(filename: str):
    # Chuẩn bị thư mục & file đích
    INPUT_DIR = Path("Input")
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = INPUT_DIR / (filename if filename.endswith(".txt") else f"{filename}.txt")
    prompt = """
Bạn là một người kiểm thử UI chuyên nghiệp.
Tạo ra chính xác 20 string mô tả kiểm thử UI khác nhau bằng tiếng Việt.

Output format (must be followed exactly):
- Return ONLY a valid JSON array of exactly 20 strings.
- Use JSON double quotes for the array items (JSON requirement).
- Inside each string, do not include any double quotes; use only single quotes for quoted labels or literals.
- Do not include numbering, markdown, code fences, or any extra text outside the JSON array.

Các ràng buộc chặt chẽ (phải tuân thủ tất cả):
- Output là 1 JSON duy nhất, chứa các câu trong [], ngăn cách nhau bởi dấu , ví dụ: ["câu 1", "câu 2",...]. Trong 1 string có thể có nhiều ý kiểm thử UI (10-15 ý) giống như ví dụ tôi để dưới.
- Tuyệt đối cấm các từ mơ hồ (ví dụ: 'phù hợp', 'nhanh', 'nhanh chóng', 'mượt mà', 'đủ', ...). Luôn đưa ra các giá trị rõ ràng, có thể đo lường được.
- Các bước phải theo trình tự và đầy đủ.
- Mỗi câu phải mô tả nhiều thuộc tính của cùng một phần tử (ví dụ: cỡ chữ, màu sắc, viền, ...).
- Mỗi hành động phải chỉ định nhiều thuộc tính cùng lúc (kiểu, bố cục, trạng thái, ...).
- Luôn cung cấp các giá trị mong đợi cụ thể, có thể đo lường được với các đơn vị hoặc số đếm khi áp dụng: kích thước pixel chính xác (ví dụ: 16px), màu RGB/HEX (ví dụ: rgb(34,139,34) hoặc #228B22 hoặc 'xanh lá cây'), thời gian (ví dụ: 800ms, 2s, 1 giây), phần trăm (ví dụ: 75%), URL (đầy đủ), chiều rộng màn hình (ví dụ: 360px), tọa độ (x,y).
- Không kết hợp hai hành động khác nhau vào cùng một câu nếu chúng không liên quan. Nếu cùng một hành động áp dụng cho hai phần tử không liên quan, hãy tách chúng thành các câu riêng để làm rõ.
- Bên trong mỗi chuỗi mô tả, chỉ sử dụng dấu nháy đơn (') cho ký tự chữ; không sử dụng bất kỳ dấu nháy kép (") nào bên trong nội dung chuỗi.

Phạm vi bao phủ (đảm bảo sự đa dạng trên 20 mục):
- Bạn chỉ được phép sử dụng các thuộc tính sau đây, và không được dùng thuộc tính nào khác:
 - Kiểu dáng: màu (color), màu nền (background color), cỡ chữ (font size), phông chữ (font family), độ đậm chữ (font weight), viền (border px), bo góc (border radius), độ mờ/độ trong suốt (opacity), khoảng đệm (padding), lề (margin)
 - Trạng thái & Tương tác: hiển thị/ẩn (visible/hidden), được focus (focused), bật/tắt (enabled/disabled), đang được chọn (active/selected)
 - Vị trí & Bố cục: tọa độ x, y, chiều rộng (width), chiều cao (height), căn chỉnh, trái (left), phải (right), trên (top), dưới (bottom)
 - Đa phương tiện: tắt tiếng (muted), toàn màn hình (fullscreen)
 - Dynamics: chuyển tiếp (transition), hoạt ảnh (animation), cuộn (scroll-top/scroll-left)
 - Văn bản: văn bản (text), placeholder, văn bản thay thế (alt text), căn chỉnh văn bản (text align)
 - Dữ liệu: số lượng (count), giá trị (value), ngày tháng (date)
 - Hình ảnh: nguồn (source), kích thước hiển thị (rendered dimensions), kích thước gốc (natural dimensions), đã tải (is loaded), thời gian tải (load time), watermark
 - Accessibility: nhãn aria-label, nhãn (label)
Mỗi trong số 20 string mô tả phải bao gồm một hoặc nhiều thuộc tính trên. Không tạo ra bất kỳ thuộc tính nào ngoài danh sách này.
Mỗi mô tả CHỈ được sử dụng các thuộc tính trong whitelist dưới đây:
Whitelist các thuộc tính được phép:
1. màu (color)
2. màu nền (background color)
3. cỡ chữ (font size)
4. phông chữ (font family)
5. độ đậm chữ (font weight)
6. viền (border (px))
7. bo góc (border radius)
8. độ mờ (opacity)
9. khoảng đệm trong (padding)
10. lề ngoài (margin)
11. căn chỉnh văn bản (text align)
12. hiển thị/ẩn (visible/hidden)
13. được focus (focused)
14. bật/tắt (enabled/disabled)
15. đang được chọn (active/selected)
16. x
17. y
18. chiều rộng (width)
19. chiều cao (height)
20. căn chỉnh
21. trái (left)
22. phải (right)
23. trên (top)
24. dưới (bottom)
25. tắt tiếng (muted)
26. toàn màn hình (fullscreen)
27. chuyển tiếp (transition)
28. hoạt ảnh (animation)
29. cuộn (scroll)
30. văn bản (text)
31. placeholder (placeholder)
32. văn bản thay thế (alt text)
33. số lượng (count)
34. giá trị (value)
35. ngày (date)
36. nguồn (source)
37. kích thước hiển thị (rendered dimensions)
38. kích thước gốc (natural dimensions)
39. đã tải (is loaded)
40. thời gian tải (load time)
41. watermark (watermark)
42. nhãn aria (aria-label)
43. nhãn (label)
44. hình dạng (shape)
Định dạng đầu ra (phải tuân thủ chính xác):
- CHỈ TRẢ VỀ một mảng JSON hợp lệ gồm chính xác 20 string.
- Sử dụng dấu nháy kép JSON cho các mục trong mảng (yêu cầu của JSON).
- Bên trong mỗi chuỗi, không bao gồm bất kỳ dấu nháy kép nào; chỉ sử dụng dấu nháy đơn cho các nhãn hoặc ký tự chữ được trích dẫn.
- Không bao gồm số thứ tự, markdown, các code fences, hoặc bất kỳ văn bản bổ sung nào bên ngoài mảng JSON.

Để giúp mô hình hiểu tổng quát hơn, hãy đa dạng cách diễn đạt cho các thuộc tính CSS bằng các từ đồng nghĩa, nhưng vẫn giữ ý nghĩa rõ ràng và có thể ánh xạ được về đúng thuộc tính trong whitelist.
Các phần tử có thể là tiếng Anh hoặc tiếng Việt, ví dụ: nút 'Submit', hình ảnh 'product_image.jpg', tiêu đề 'Happy Holiday'. Nhưng lưu ý chỉ dùng tiếng Anh đối với tên, còn lại vẫn phải miêu tả bằng tiếng Việt.
Ví dụ:
- color có thể miêu tả bằng: 'màu chữ', 'màu văn bản', 'màu hiển thị', 'màu của tiêu đề', 'màu biểu tượng'
- background color: 'màu nền', 'màu phía sau', 'nền hiển thị', 'phần nền của nút'
- font size: 'kích thước chữ', 'độ lớn phông', 'cỡ chữ hiển thị', 'độ cao ký tự'
- font family: 'kiểu chữ', 'phông chữ', 'bộ font', 'dạng chữ hiển thị'
- border radius: 'bo góc', 'độ cong góc', 'viền bo tròn'
- box shadow: 'đổ bóng', 'bóng của khối', 'hiệu ứng bóng', 'bóng mờ phía sau'
- alignment: 'căn chỉnh', 'vị trí canh giữa', 'canh lề', 'căn trái/phải/giữa'
- width/height: 'chiều rộng', 'bề ngang', 'kích thước ngang'; 'chiều cao', 'độ cao hiển thị'
- opacity: 'độ trong suốt', 'độ mờ', 'độ hiển thị rõ'
- padding/margin: 'khoảng cách trong', 'khoảng đệm', 'lề ngoài', 'khoảng trống xung quanh'

Yêu cầu khi sử dụng từ đồng nghĩa:
- Không được làm thay đổi nghĩa gốc của thuộc tính.
- Mỗi mô tả kiểm thử nên xen kẽ các từ đồng nghĩa khác nhau cho cùng loại thuộc tính để tăng tính đa dạng ngôn ngữ.
- Tuy nhiên, trong JSON đầu ra, chỉ cần biểu diễn dưới dạng chuỗi văn bản (không cần ánh xạ ngược).
- Mục tiêu là giúp mô hình học cách nhận biết rằng nhiều cụm từ khác nhau đều mô tả cùng một đặc tính UI.

Các từ mô tả nên tự nhiên như ngôn ngữ con người, không rập khuôn “color = màu”, nhưng vẫn phải rõ ràng và dễ hiểu.
Ví dụ hợp lệ: “Kiểm tra rằng tiêu đề có màu chữ trắng và nền xanh nhạt, bo góc nhẹ 6px, chữ căn giữa.”
Không sử dụng bừa bãi dấu ' hoặc dấu ngoặc kép ". Chỉ dùng dấu ' khi cần trích dẫn link, văn bản.

Ví dụ 1 string (không được copy):
"Mở trang 'https://www.shopdemo.com/product/123'. Xác minh rằng hình ảnh sản phẩm có nguồn là 'hero_image.jpg' và kích thước hiển thị là 400x300px. Đảm bảo hình ảnh có alt text chứa 'Giày chính hãng'. Kiểm tra rằng chính xác 6 hình thu nhỏ được hiển thị phía dưới với căn chỉnh ở giữa và mỗi hình thu nhỏ có chiều rộng 120px và chiều cao 80px. Xác nhận tiêu đề chính hiển thị văn bản chính xác 'Tiêu đề sản phẩm' với cỡ chữ 18px và màu #000000. Xác minh rằng nút 'Mua ngay' hiển thị và chuyển sang trạng thái được chọn với màu nền #ff0000. Xác nhận nút có bán kính viền là 6px, cuộn trang và xác minh tiêu đề vẫn cố định (sử dụng hành vi cuộn). Kiểm tra nhãn ngôn ngữ có aria-label được đặt thành English."
Mỗi string cần có nhiều hành động kiểm thử như ví dụ trên (10-15 hành động), với các thuộc tính khác nhau từ whitelist.
Sinh 20 string, rồi cho vào mảng JSON trả về. Trả về đúng định dạng yêu cầu (các câu trong [], ngăn cách nhau bởi dấu , ví dụ: ["câu 1", "câu 2",...]).
"""

    all_sentences = []
    for _ in range(5):  # 5 batch -> ~100 câu
        raw = call_gemini(prompt)
        text = extract_json_from_text(raw) if raw else ""
        sentences = []
        # 1) cố parse như JSON array
        try:
            data = json.loads(text)
            if isinstance(data, list):
                sentences = [str(s) for s in data]
        except Exception as e:
            print(f"❌ Error parsing JSON: {e}")
            pass
        # 2) nếu không phải JSON array, tách theo dòng (đơn giản, không kiểm lỗi/chính tả)
        if not sentences:
            sentences = [ln.strip().strip(",") for ln in text.splitlines() if ln.strip()]
        # Chuẩn hoá: thay tất cả dấu " bên trong câu thành ' ; bọc toàn bộ câu trong dấu "
        for s in sentences:
            s = s.replace('"', "'").strip()
            if not (s.startswith('"') and s.endswith('"')):
                s = f'"{s}"'
            all_sentences.append(s)

    # Ghi file: các câu cách nhau bởi dấu phẩy, vẫn giữ mỗi câu trong dấu "
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
        prompt = f"""Bạn là một trình tạo trường hợp kiểm thử UI chuyên nghiệp.
Nhiệm vụ của bạn là chuyển đổi mô tả kiểm thử ngôn ngữ tự nhiên sau thành một danh sách các hành động UI rõ ràng, nguyên tử.
Hãy tuân thủ các quy tắc nghiêm ngặt sau:
- Đầu ra phải là một mảng JSON của các mảng.
- Nhóm các thuộc tính liên quan của cùng một phần tử UI trong cùng một trạng thái vào một tác vụ nguyên tử.
  - Điều này bao gồm tất cả các thuộc tính trực quan, bố cục, định dạng, kiểu, căn chỉnh và nội dung (ví dụ: cỡ chữ, màu sắc, nền, bán kính viền, căn chỉnh).
  - Không tách văn bản thay thế (alt text), hình mờ (watermark), kích thước tự nhiên so với kích thước kết xuất, độ sắc nét/độ nén, v.v. - nếu tất cả đều đề cập đến cùng một hình ảnh ở trạng thái bình thường của nó.
  - Không tách các thứ như màu sắc, định dạng, căn chỉnh, phông chữ, lề trong (padding), kích thước, vị trí, so sánh với một phần tử khác nếu chúng thuộc cùng một phần tử trong cùng một ngữ cảnh.
  - (Ví dụ: Nút 'Thêm vào giỏ hàng' có màu '#ff9900', bán kính viền '4px' và màu nền chuyển thành '#e68a00' khi được chọn)
  - Chỉ tách các tác vụ nếu:
    - Các thuộc tính liên quan đến các trạng thái khác nhau (ví dụ: khi di chuột so với bình thường).
    - Các thuộc tính liên quan đến các phần tử khác nhau.
    - Hướng dẫn liên quan đến một hành động của người dùng theo sau là một xác minh (ví dụ: nhấp rồi kiểm tra, định vị rồi kiểm tra, định vị rồi gõ), hoặc một hành động của người dùng rõ ràng ở giữa là bắt buộc.
    - Hướng dẫn liên quan đến nhiều hành động (ví dụ: "Định vị trường nhập liệu 'Email' và nhập 'invalid-email'" phải được tách thành 2 tác vụ: Định vị trường nhập liệu 'Email', Nhập 'invalid-email' vào trường nhập liệu 'Email').
    - Hành vi khác nhau giữa các loại thiết bị (ví dụ: di động so với máy tính để bàn).
    - Hành động yêu cầu nhiều bước để hoàn thành (ví dụ: Đăng nhập bằng tài khoản 'patient001' và mật khẩu 'Health@2023' phải được tách thành 2 tác vụ: Nhập 'patient001' vào trường tài khoản, Nhập 'Health@2023' vào trường mật khẩu).
- Mỗi mục trong mảng là một chuỗi hành động kiểm thử nguyên tử duy nhất.
- Không tạo các bước không được mô tả trong tác vụ chính.
KHÔNG bao gồm giải thích, markdown, bình luận hoặc bất cứ thứ gì khác - chỉ mảng JSON.
Không sử dụng bừa bãi dấu ' hoặc dấu ngoặc kép ". Chỉ dùng dấu ' khi cần trích dẫn link, văn bản, text.
Ví dụ:
- Requirement:
[
"Mở 'https://www.netflix.com'. Nhấp vào nút 'Đăng nhập' nằm ở góc trên cùng bên phải (cách mép trên 50px và mép phải 30px). Nhập email 'testuser@example.com' và mật khẩu 'Test@1234'. Gửi biểu mẫu đăng nhập. Xác minh rằng sau khi gửi, người dùng được chuyển hướng đến 'https://www.netflix.com/browse' trong vòng 3 giây và trang hiển thị ít nhất 5 hình thu nhỏ phim được cá nhân hóa, và đảm bảo rằng tất cả các trường nhập liệu có phông chữ Roboto 16px, lề trong 12px, lề dưới 16px. Nút 'Đăng nhập' có màu nền #e50914, màu chữ #ffffff, bán kính viền 4px, có khả năng phản hồi trên màn hình rộng 360px và tất cả các phần tử đều đáp ứng tỷ lệ tương phản WCAG ít nhất 4.5:1.",
"Mở 'https://www.amazon.com'. Tìm kiếm 'Dàn loa Bluetooth', xác minh rằng tìm kiếm hoàn thành trong vòng dưới 2 giây, hiển thị ít nhất 10 sản phẩm với tiêu đề chứa 'Loa Bluetooth'. Xác nhận mỗi thẻ sản phẩm có chiều rộng hình ảnh chính xác 150px, giá sản phẩm có cỡ chữ 18px, và đảm bảo nút 'Thêm vào giỏ hàng' có màu #ff9900, các góc bo tròn 4px, và màu nền khi di chuột thay đổi thành #e68a00.",
"Mở 'https://shop.example.com/item/P123', xác minh rằng hình ảnh sản phẩm chính '#main-product-image' tải thành công trong vòng 800ms, có văn bản thay thế chứa 'Ví da Premium', không có hình mờ, kích thước tự nhiên 1600x1200 px, hiển thị ở 400x300px với dung sai ±4px, độ sắc nét ≥ 0.85, hiện vật nén ≤ 0.1, và đảm bảo hình ảnh được định vị ở bên trái của khối giá '.price' với khoảng cách tối thiểu 24px (±2px)."
]
- Response:
[
  [
"Mở 'https://www.netflix.com'",
"Định vị nút 'Đăng nhập' ở góc trên cùng bên phải (cách mép trên 50px và mép phải 30px)",
"Nhấp vào nút 'Đăng nhập'",
"Nhập 'testuser@example.com' vào trường nhập email",
"Nhập 'Test@1234' vào trường nhập mật khẩu",
"Gửi biểu mẫu đăng nhập",
"Xác minh rằng sau khi gửi, người dùng được chuyển hướng đến 'https://www.netflix.com/browse' trong vòng 3 giây",
"Xác minh rằng trang hiển thị ít nhất 5 hình thu nhỏ phim được cá nhân hóa",
"Xác minh rằng tất cả các trường nhập liệu có phông chữ Roboto 16px, lề trong 12px, lề dưới 16px",
"Xác minh rằng nút 'Đăng nhập' có màu nền #e50914, màu chữ #ffffff, bán kính viền 4px",
"Xác minh rằng nút 'Đăng nhập' có khả năng phản hồi trên màn hình rộng 360px",
"Xác minh rằng tất cả các phần tử đều đáp ứng tỷ lệ tương phản WCAG ít nhất 4.5:1"
  ],
  [
"Mở 'https://www.amazon.com'",
"Tìm kiếm 'Dàn loa Bluetooth'",
"Xác minh rằng tìm kiếm hoàn thành trong vòng dưới 2 giây",
"Xác minh rằng kết quả tìm kiếm hiển thị ít nhất 10 sản phẩm với tiêu đề chứa 'Loa Bluetooth'",
"Xác minh rằng mỗi thẻ sản phẩm có chiều rộng hình ảnh chính xác 150px",
"Xác minh rằng giá sản phẩm có cỡ chữ 18px",
"Xác minh rằng nút 'Thêm vào giỏ hàng' có màu #ff9900 và các góc bo tròn 4px",
"Xác minh rằng màu nền của nút 'Thêm vào giỏ hàng' thay đổi thành #e68a00 khi di chuột"
  ],
  [
"Mở 'https://shop.example.com/item/P123'",
"Xác minh rằng hình ảnh sản phẩm chính '#main-product-image' tải thành công trong vòng 800ms, có văn bản thay thế chứa 'Ví da Premium', không có hình mờ, kích thước tự nhiên 1600x1200 px, hiển thị ở 400x300px với dung sai ±4px, độ sắc nét ≥ 0.85, hiện vật nén ≤ 0.1, và đảm bảo hình ảnh được định vị ở bên trái của khối giá '.price' với khoảng cách tối thiểu 24px (±2px)"
  ]
]
Bây giờ hãy tạo JSON từ yêu cầu này:

- Requirement:
{chr(10).join([f'{i+1}. "{t}"' for i, t in enumerate(batch)])}

Bây giờ chỉ trả về mảng JSON của các mảng, theo cùng một thứ tự.
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
Your task is to convert the following Vietnamese atomic test instruction into a list of executable UI test steps in JSON format (must in English).

Instruction: "{subtask_list}"

Each step must follow this JSON structure:
{{
  "action": "action to perform (e.g. click, type, hover, locate, verify, etc.)",
  "selector": "CSS selector or a natural description (keep Vietnamese text/labels if present in the instruction, e.g. 'Đăng nhập' button), or empty string if not applicable",
  "value": "string value to type or expect, or empty string if not applicable, keep Vietnamese text if provided",
  "expected": {{
    // This object may contain multiple properties,
    // based only on what the instruction requires.
    // Each key is the name of a UI property (e.g. color, font, position).
    // Each value is the expected value.
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
  - 'text':
  - 'placeholder':
  - 'language': e.g., `"French"`, `"Chinese"`
  - 'position': 'relative', 'absolute', 'fixed', 'static',...
  - 'x', 'y', 'width', 'height': px (e.g.,{{"x": 100, "y": 200, "width": "300px", "height": "150px"}})
  - 'overflow': overflow-x, overflow-y, overflow-block, overflow-inline (e.g., "hidden", "scroll", "visible", "auto")
  - 'occluded': boolean (true if element is occluded by another, false if not)
  - 'text-align':
  - `color`: e.g., "#ffffff" or "rgb(255, 255, 255)" or "white"
  - `background-color`
  - `font-family`, `font-size`, `font-weight`, `font-style` if mentioned
  - 'border': width, style, color, radius
  - 'padding': top, right, bottom, left
  - 'margin': top, right, bottom, left
  - 'text-align': "center", "left", "right", "justify",...
  - 'line-height': e.g., "1.5", "20px"
  - 'vertical-align': "middle", "top", "bottom", 'baseline',...
  - 'align-items': "center", "flex-start", "flex-end", 'space-between',...
  - 'justify-content': "space-between", 'center', "flex-end", "flex-start",...
  - 'align-content': "flex-start", "center", "space-around",...
  - 'align-self': "stretch", "center", "flex-start",...
  - 'gap': e.g., "10px"
  - 'opacity': "0.5", "1", etc.
  - Or any property related to styles, as long as it follows the correct format and rules stated (only declare if present in the input description).
  - 'visibility': "visible" or "hidden"
  - 'enabled': boolean (true or false)
  - 'focused': boolean
  - 'hovered': boolean     
  - 'active': boolean
  - Or any property related to state, as long as it follows the correct format and rules stated (only declare if present in the input description).   
  - dynamics:
  - `movement`: "static" or "moves" after an action
  - `scroll`: "sticky" | "fixed" | "static" | "none" or something
  - 'parallax': number (e.g., `0.5` for parallax effect)
  - `transition`: CSS transition
  - 'animation': CSS animation
  - `focus`: boolean, ('true' if element should be focused, via Tab or click)
  - `navigation`: destination URL if checking page redirection
  - `loading`: `"present", "loading", "none"` for spinners, indicators
  - 'progress-value': string, (% value if it is progress bar, pie chart, etc)
  - Or any property related to 'dynamics', as long as it follows the correct format and rules stated (only declare if present in the input description).
  - If the element is image (img), its may contain:
  - `src`: URL of the image
  - 'alt-text': string (e.g., "Hình ảnh sản phẩm", "Logo")
  - 'is-loaded': boolean (true if image is loaded, false if not)
  - 'load-time': number (in milliseconds, e.g., 500 for 0.5 seconds)
  - 'sharpness-score-gte': number (0 to 1, e.g., 0.8 for sharpness)
  - 'compression-artifacts-lte': number (0 to 1, e.g., 0.2 for compression artifacts)
  - 'rendered-dimensions': `width`, `height`, 'tolerance-px' (in pixels, e.g., 300x200)
  - 'natural-dimensions': `width`, `height` (natural size of the image)
  - 'viewport-dimensions': `width`, `height` (size of the image in the viewport)
  - 'watermark': boolean (true if image has a watermark, false if not)
  - Or any property related to expected , as long as it follows the correct format and rules stated (only declare if present in the input description).

I am using Selenium with ChromeDriver to extract the UI (User Interface) information of web elements.  
Therefore, all attributes must follow these rules:
1. Only include CSS properties that can be retrieved through `window.getComputedStyle(element)`.
2. The property names must follow Chrome’s computed style format:
   - All lowercase letters.
   - Words separated by hyphens (-).
   - Example: background-color, font-weight, border-radius, box-shadow.
3. Do NOT use camelCase or PascalCase.

Example 1:
- Task list:
[
  "Mở 'https://www.amazon.com'",
  "Tìm kiếm 'Dàn loa Bluetooth'",
  "Xác minh rằng tìm kiếm hoàn thành trong vòng dưới 2 giây",
  "Xác minh rằng kết quả tìm kiếm hiển thị ít nhất 10 sản phẩm với tiêu đề chứa từ 'Loa Bluetooth'",
  "Xác minh rằng mỗi thẻ sản phẩm có chiều rộng hình ảnh chính chính xác 150px",
  "Xác minh rằng nút 'Thêm vào giỏ hàng' có màu #ff9900 và bán kính viền 4px, được định vị tại x=120px và y=300px, với chiều rộng 200px và chiều cao 50px",
  "Xác minh rằng màu nền của nút 'Thêm vào giỏ hàng' thay đổi thành #e68a00 khi di chuột"
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
    "value": "Dàn loa Bluetooth",
    "expected": {{
    }}
  }},
  {{
    "action": "verify",
    "selector": "search results",
    "value": "",
    "expected": {{
      "visibility": "visible",
      "duration": 2000
    }}
  }},
  {{
    "action": "verify",
    "selector": ".search-results .product-title",
    "value": "",
    "expected": {{
      "count": 10,
      "text-content": Loa Bluetooth"
    }}
  }},
  {{
    "action": "verify",
    "selector": "'product card'",
    "value": "",
    "expected": {{
      "width": "150px"
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Thêm vào giỏ hàng' button",
    "value": "",
    "expected": {{
      "text-content": "Thêm vào giỏ hàng",
      "color": "#ff9900",
      "border-radius": "4px"
      "x": "120px",
      "y": "300px",
      "width": "200px",
      "height": "50px"
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Thêm vào giỏ hàng' button",
    "value": "",
    "expected": {{
      "text": "Thêm vào giỏ hàng",
      "hovered": true,
      "background-color": "#e68a00"
    }}
  }}
]

Example 2:
- Task list:
[
  "Định vị nút 'Đăng nhập' ở góc trên cùng bên phải (cách mép trên 50px và mép phải 30px)",
  "Nhấp vào nút 'Đăng nhập'",
  "Nhập 'testuser@example.com' vào trường nhập email",
  "Nhập 'Test@1234' vào trường nhập mật khẩu",
  "Gửi biểu mẫu đăng nhập",
  "Xác minh rằng sau khi gửi, người dùng được chuyển hướng đến 'https://www.netflix.com/browse' trong vòng 3 giây",
  "Xác minh rằng trang hiển thị ít nhất 5 hình thu nhỏ phim được cá nhân hóa",
  "Xác minh rằng tất cả các trường nhập liệu có phông chữ Roboto 16px, lề trong 12px và lề dưới 16px",
  "Xác minh rằng nút 'Đăng nhập' có màu nền #e50914, màu chữ #ffffff và bán kính viền 4px",
  "Xác minh rằng nút 'Đăng nhập' có khả năng phản hồi trên màn hình rộng 360px",
  "Xác minh rằng tất cả các phần tử hiển thị đều đáp ứng tỷ lệ tương phản WCAG ít nhất 4.5:1"
]
- Response:
[
  {{
    "action": "locate",
    "selector": "'Đăng nhập' button",
    "value": "",
    "expected": {{
      "text-content": "Đăng nhập",
      "top": "50px",
      "right": "30px"
    }}
  }},
  {{
    "action": "click",
    "selector": "'Đăng nhập' button",
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
    "selector": "'Đăng nhập' button",
    "value": "",
    "expected": {{
      "text-content": "Đăng nhập",
      "background-color": "#e50914",
      "color": "#ffffff",
      "border-radius": "4px"
    }}
  }},
  {{
    "action": "verify",
    "selector": "'Đăng nhập' button",
    "value": "",
    "expected": {{
      "text-content": "Đăng nhập",
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
  "Mở 'https://shop.example.com/item/P123'",
  "Xác minh rằng hình ảnh sản phẩm chính '#main-product-image' tải thành công trong vòng 800ms, có văn bản thay thế chứa 'Premium Leather Wallet', không có hình mờ, kích thước tự nhiên 1600x1200, hiển thị ở 400x300, độ sắc nét ≥ 0.85 và các hiện vật nén ≤ 0.1"
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
      "loadTime": 800,
      "alt-text": "Premium Leather Wallet",
      "watermark": false,
      "natural-width": 1600,
      "natural-height": 1200,
      "rendered-width": 400,
      "rendered-height": 300,
      "shapness-score": 0.85,
      "compression-artifacts-lte": 0.1
    }}
  }}
]

Example 4:
- Task list:
[
  "Nhấp vào liên hệ 'Nguyen Van An'",
  "Nhập 'Hello' vào trường nhập tin nhắn có chiều cao 40px và cỡ chữ 14px",
  "Gửi tin nhắn",
  "Xác minh rằng tin nhắn đã gửi xuất hiện trong cửa sổ trò chuyện trong vòng 1 giây, với màu nền #dcf8c6, căn lề phải, và thời gian của tin nhắn được định dạng theo kiểu '12-hour', căn lề ở góc dưới cùng bên phải bên trong cửa sổ trò chuyện"
]

- Response:
[
  {{
    "action": "click",
    "selector": "'Nguyen Van An' contact",
    "value": "",
    "expected": {{
    }}
  }},
  {{
    "action": "type",
    "selector": "'message' input",
    "value": "Hello",
    "expected": {{
      "height": "40px",
      "font-size": "14px"
    }}
  }},
  {{
    "action": "submit",
    "selector": "'send message' button",
    "value": "",
    "expected": {{
    }}
  }},
  {{
    "action": "verify",
    "selector": "'sent message bubble'",
    "value": "",
    "expected": {{
      "visibility": visible,
      "text-align": "right",
      "background-color": "#dcf8c6",
      "timeout": 1000,
      "timestamp": "12-hour",
      "display": "flex",
      "justify-content": "flex-end",
      "align-items": "flex-end"
    }}
  }}
]

Notes:
- Each atomic test instruction return only ONE test steps, NO splitting the instruction into multiple substeps.
- The selector can be a CSS selector or a natural-language reference to the element.
- UI may be implemented using any frontend framework (e.g., TailwindCSS, Bootstrap, raw HTML).
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
                    step_map.append((subtask_list[idx], [step_obj]))

                all_step_groups.append(normalized)
                break

            except Exception as e:
                print(f"❌ Error (attempt {attempt + 1}/3): {e}")
                print(f"⚠️ Raw response: {response}")
                if attempt == retries - 1:
                    # Fallback: tạo step rỗng giữ chỗ (để pipeline không gãy)
                    fallback = [{
                        "action": "",
                        "selector": "",
                        "value": "",
                        "expected": {}
                    } for _ in subtask_list]
                    all_step_groups.append(fallback)
                    for idx, st in enumerate(subtask_list):
                        step_map.append((st, [fallback[idx]]))
        # Sau khi xử lý xong, khôi phục về cấu trúc song song task_trace
    step_trace = step_map
    step_groups_flat = [steps for _, steps in step_map]

# ✅ Gộp lại theo cấu trúc gốc ban đầu (đảm bảo đúng hàng Excel)
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
            
            print(f"\n▶️ Running pipeline for: {case_id}")
            task_file = INPUT_DIR / f"{case_id}.txt"
            task_json_file = TASK_DIR / f"{case_id}.task.json"
            task_list, task_trace = step1_analyze_task(task_file)
            save_json(task_json_file, task_list)
            print("\n Step 1 success \n")  

            step_json_file = STEP_DIR / f"{case_id}.step.json"
            step_groups, step_trace = step2_generate_steps(task_list)
            save_json(step_json_file, step_groups)
            print("\n Step 2 success \n")

            report_file = REPORT_DIR / f"{case_id}.summary.xlsx"
            save_excel_summary(report_file, task_trace, step_groups)
            print(f"✅ Excel saved to: {report_file}")
        except Exception as e:
            print(f"❌ Error in {case_id}: {e}")

if __name__ == "__main__":
    main()
