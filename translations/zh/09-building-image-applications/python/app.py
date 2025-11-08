import base64
import json
import mimetypes
import os
import dashscope
import dotenv
import requests
from PIL import Image
from dashscope import MultiModalConversation

dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'

messages = [
    {
        "role": "user",
        "content": [
            {"text": "Bunny on horse, holding a lollipop, on a foggy meadow where it grows daffodils'"}
        ]
    }
]

# 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY","")
assert api_key, "ERROR: OpenAI Key is missing"

response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-plus",
    messages=messages,
    result_format='message',
    stream=False,
    watermark=False,
    prompt_extend=True,
    negative_prompt='',
    size='1328*1328'
)

if response.status_code == 200:
    print(json.dumps(response, ensure_ascii=False))
    # Set the directory for the stored image
    image_dir = os.path.join(os.curdir, 'images')

    # If the directory doesn't exist, create it
    if not os.path.isdir(image_dir):
        os.mkdir(image_dir)

    # Initialize the image path (note the filetype should be png)
    image_path = os.path.join(image_dir, 'generated-image.png')

    image_url = response.output.choices[0].message.content[0]['image'] # extract image URL from response
    generated_image = requests.get(image_url).content  # download the image
    with open(image_path, "wb") as image_file:
        image_file.write(generated_image)

    # Display the image in the default image viewer
    image = Image.open(image_path)
    image.show()
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print("请参考文档：https://www.alibabacloud.com/help/zh/model-studio/error-code")

# ---用于 Base64 编码 ---
# 格式为 data:{mime_type};base64,{base64_data}
def encode_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError("不支持或无法识别的图像格式")

    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(
                image_file.read()).decode('utf-8')
        return f"data:{mime_type};base64,{encoded_string}"
    except IOError as e:
        raise IOError(f"读取文件时出错: {file_path}, 错误: {str(e)}")

def download_image(image_url, save_name='output.png'):
    try:
        response = requests.get(image_url, stream=True, timeout=300)  # 设置超时
        response.raise_for_status()  # 如果HTTP状态码不是200，则引发异常
        with open(os.path.join(image_dir, save_name), 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"图像已成功下载到: {os.path.join(image_dir, save_name)}")
        return os.path.join(image_dir, save_name)
    except requests.exceptions.RequestException as e:
        print(f"图像下载失败: {e}")

# 获取图像的 Base64 编码
# 调用编码函数，请将 "/path/to/your/image.png" 替换为您的本地图片文件路径，否则无法运行
image = encode_file(image_path)

messages = [
    {
        "role": "user",
        "content": [
            {"image": image},
            {"text": "An image of a rabbit with a hat on its head."}]
    }
]

# 模型仅支持单轮对话，复用了多轮对话的接口
# qwen-image-edit-plus支持输出1-6张图片，此处以2张为例
response = MultiModalConversation.call(
    api_key=api_key,
    model="qwen-image-edit-plus",
    messages=messages,
    stream=False,
    n=2,
    watermark=False,
    negative_prompt=" "
)

if response.status_code == 200:
    # 如需查看完整响应，请取消下行注释
    # print(json.dumps(response, ensure_ascii=False))
    for i, content in enumerate(response.output.choices[0].message.content):
        print(f"输出图像{i+1}的URL:{content['image']}")
        image_path = download_image(content['image'])
        # Display the image in the default image viewer
        image = Image.open(image_path)
        image.show()
else:
    print(f"HTTP返回码：{response.status_code}")
    print(f"错误码：{response.code}")
    print(f"错误信息：{response.message}")
    print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")