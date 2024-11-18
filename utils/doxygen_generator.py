import os
import re
import argparse
import subprocess
import logging
import webbrowser
import shutil
import time
from string import Template

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def wait_for_file(filepath, timeout=120):
    """轮询等待文件生成，直到超时"""
    start_time = time.time()
    while not os.path.exists(filepath):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"等待文件 {filepath} 超时")
        time.sleep(1)  # 每秒检查一次文件是否存在


def extract_project_info(startpath):
    """尝试从 main.cpp 提取项目名称和简要说明"""
    main_file = os.path.join(startpath, "main.cpp")
    project_name = "Default Project Name"
    project_brief = "Default Project Brief"
    if os.path.exists(main_file):
        with open(main_file, "r", encoding="utf-8") as file:
            content = file.read()
            mainpage_match = re.search(
                r"@mainpage\s+([^\n]+)\s*\*\s*\n\s*\*\s+(.+?)(?:\n\s*\*|$)",
                content,
                re.DOTALL,
            )
            if mainpage_match:
                project_name = mainpage_match.group(1).strip()
                project_brief = mainpage_match.group(2).strip()
    return project_name, project_brief


def clean_output_dir(output_dir):
    """清理输出目录中的旧生成文件"""
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        logging.info(f"已清理输出目录: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)


def copy_file_safe(src, dst):
    """安全复制文件，处理找不到文件的情况，并处理同名文件"""
    try:
        if os.path.exists(dst):
            logging.info(f"目标文件 {dst} 已存在，将被覆盖")
        shutil.copy2(src, dst)
        logging.info(f"已复制文件 {src} 到 {dst}")
    except FileNotFoundError:
        logging.error(f"无法找到文件: {src}")
    except Exception as e:
        logging.error(f"复制文件时出错: {e}")


# def create_readme_redirect(parent_dir, html_dir):
#     """创建 readme.html 并自动重定向到 index.html"""
#     readme_path = os.path.join(parent_dir, "readme.html")
#     relative_html_path = os.path.relpath(
#         os.path.join(html_dir, "index.html"), parent_dir
#     )

#     # 确保使用正斜杠作为路径分隔符
#     relative_html_path = relative_html_path.replace("\\", "/")

#     # 创建 readme.html，包含重定向逻辑
#     with open(readme_path, "w", encoding="utf-8") as readme:
#         readme.write(
#             f"""<!DOCTYPE html>
# <html lang="en">
# <head>
#     <meta http-equiv="refresh" content="0; url=./{relative_html_path}" />
#     <meta charset="UTF-8">
#     <meta name="viewport" content="width=device-width, initial-scale=1.0">
#     <title>Redirecting to Documentation</title>
# </head>
# <body>
#     <p>If you are not redirected automatically, follow this <a href="./{relative_html_path}">link to the documentation</a>.</p>
# </body>
# </html>
# """
#         )
#     logging.info(f"readme.html 已在 {parent_dir} 中创建，带有自动重定向功能")


def create_doxygen_html(startpath):
    # 定义路径
    parent_dir = os.path.dirname(startpath)
    html_dir = os.path.join(parent_dir, "doxygen_output", "html")
    index_html_path = os.path.join(html_dir, "index.html")
    doxygen_html_path = os.path.join(parent_dir, "Doxygen.html")

    # 读取 index.html 内容
    if os.path.exists(index_html_path):
        with open(index_html_path, "r", encoding="utf-8") as file:
            content = file.read()

        # 添加 <base> 标签
        base_tag = '<base href="./doxygen_output/html/" />\n'
        content = content.replace("<head>", f"<head>\n{base_tag}", 1)

        # 写入 Doxygen.html
        with open(doxygen_html_path, "w", encoding="utf-8") as file:
            file.write(content)
        logging.info(f"已创建 {doxygen_html_path}")

        # 自动打开 Doxygen.html
        webbrowser.open_new_tab(doxygen_html_path)
        logging.info(f"已打开 {doxygen_html_path}")
    else:
        logging.error(f"无法找到 {index_html_path}")


def generate_doxygen(startpath, project_name=None):
    """生成 Doxygen 配置文件并运行 Doxygen"""
    if not project_name:
        project_name, project_brief = extract_project_info(startpath)
    else:
        _, project_brief = extract_project_info(startpath)

    # 设置输出目录到 src 的父目录
    output_dir = os.path.join(os.path.dirname(startpath), "doxygen_output")
    html_dir = os.path.join(output_dir, "html")
    latex_dir = os.path.join(output_dir, "latex")

    clean_output_dir(output_dir)

    doxyfile_path = os.path.join(startpath, "Doxyfile")
    template_path = os.path.join(os.path.dirname(__file__), "Doxyfile_template")

    # 读取Doxyfile模板
    with open(template_path, "r", encoding="utf-8") as template_file:
        doxy_template = Template(template_file.read())

    # 填充模板
    doxy_content = doxy_template.substitute(
        PROJECT_NAME=project_name,
        PROJECT_BRIEF=project_brief,
        MAIN_CPP_PATH=os.path.join(startpath, "main.cpp"),
        START_PATH=startpath,
        OUTPUT_DIR=os.path.abspath(output_dir),
    )

    # 创建 doxygen-awesome-css 目录并复制 CSS 和 JS 文件
    css_target_dir = os.path.join(startpath, "doxygen-awesome-css")
    os.makedirs(css_target_dir, exist_ok=True)
    resource_files = [
        "doxygen-awesome.css",
        "doxygen-awesome-sidebar-only.css",
        "doxygen-awesome-sidebar-only-darkmode-toggle.css",
        "doxygen-awesome-darkmode-toggle.js",
        "doxygen-awesome-paragraph-link.js",
        "doxygen-awesome-interactive-toc.js",
    ]
    for resource_file in resource_files:
        copy_file_safe(
            os.path.join(
                os.path.dirname(__file__), "doxygen-awesome-css", resource_file
            ),
            os.path.join(css_target_dir, resource_file),
        )

    # 复制 header.html,footer.html和Comphys.svg
    header_source = os.path.join(os.path.dirname(__file__), "header.html")
    footer_source = os.path.join(os.path.dirname(__file__), "footer.html")
    svg_source = os.path.join(os.path.dirname(__file__), "comphys.svg")
    header_target = os.path.join(css_target_dir, "header.html")
    footer_target = os.path.join(css_target_dir, "footer.html")
    logo_target = os.path.join(css_target_dir, "comphys.svg")
    copy_file_safe(header_source, header_target)
    copy_file_safe(footer_source, footer_target)
    copy_file_safe(svg_source, logo_target)

    # 写入Doxyfile
    with open(doxyfile_path, "w", encoding="utf-8") as doxyfile:
        doxyfile.write(doxy_content)
    logging.info(f"Doxyfile 已创建: {doxyfile_path}")

    # 运行 Doxygen
    try:
        subprocess.run(["doxygen", doxyfile_path], check=True, cwd=startpath)
        logging.info("Doxygen 运行成功")
    except subprocess.CalledProcessError as e:
        logging.error(f"Doxygen 运行失败，错误: {e}")
        return

    # 等待 index.html 生成
    index_html_path = os.path.join(html_dir, "index.html")
    try:
        wait_for_file(index_html_path, timeout=120)
        logging.info(f"index.html 已生成: {index_html_path}")
    except TimeoutError as e:
        logging.error(str(e))
        return

    # 创建index.html的重定向base之后的Doxygen.html并打开
    create_doxygen_html(startpath)

    # 运行 make.bat 来编译 LaTeX 生成 PDF
    make_bat_path = os.path.join(latex_dir, "make.bat")
    if os.path.exists(make_bat_path):
        try:
            subprocess.run([make_bat_path], cwd=latex_dir, check=True, shell=True)
            logging.info("LaTeX 编译成功")
        except subprocess.CalledProcessError as e:
            logging.error(f"LaTeX 编译失败，错误: {e}")
            return
    else:
        logging.error(f"找不到 make.bat 文件: {make_bat_path}")
        return

    # 等待 refman.pdf 生成
    refman_pdf_path = os.path.join(latex_dir, "refman.pdf")
    try:
        wait_for_file(refman_pdf_path, timeout=300)
        logging.info(f"refman.pdf 已生成: {refman_pdf_path}")
    except TimeoutError as e:
        logging.error(str(e))
        return

    # 删除临时的 Doxyfile 和 CSS 文件夹
    os.remove(doxyfile_path)
    shutil.rmtree(css_target_dir)
    logging.info("临时文件夹和Doxyfile已删除")


# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Doxygen 文档生成器")
    parser.add_argument("startpath", help="项目起始路径")
    parser.add_argument("--project_name", help="项目名称（可选）")
    args = parser.parse_args()

    generate_doxygen(args.startpath, args.project_name)
