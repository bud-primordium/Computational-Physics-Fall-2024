# structure_printer.py
import os
import re
import argparse
import logging

# import subprocess
import pyperclip  # 如果使用 pyperclip

# 如果不使用 pyperclip，请注释掉上面的 import 并取消下面的注释
# import subprocess

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = [".h", ".cpp", ".f90"]


def extract_doxygen_comments(file_path):
    """
    提取文件中的 Doxygen 注释，并关联到相应的函数或类。

    返回一个列表，每个元素是一个字典，包含 'comment' 和 'entity'（函数或类名称）。
    """
    doxygen_comments = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        i = 0
        while i < len(lines):
            line = lines[i]
            # 检查多行 Doxygen 注释 /** ... */ 或 /*! ... */
            multi_line_match = re.match(r"/\*\*|/\*!+", line.strip())
            if multi_line_match:
                comment_lines = []
                while i < len(lines):
                    comment_line = lines[i].strip()
                    comment_lines.append(comment_line)
                    if "*/" in comment_line:
                        break
                    i += 1
                comment = "\n".join(comment_lines)
                # 提取注释内容（去掉起始的 /** 或 /*! 以及末尾的 */）
                comment_content = re.sub(
                    r"^/\*\*+|^/\*!+| \*/$", "", comment, flags=re.MULTILINE
                ).strip()

                # 寻找紧随其后的函数或类定义
                entity = find_next_entity(lines, i + 1)

                doxygen_comments.append({"comment": comment_content, "entity": entity})

            # 检查单行 Doxygen 注释 /// ...
            single_line_cpp = re.match(r"^\s*///(.*)", line)
            if single_line_cpp:
                comment_lines = []
                while i < len(lines):
                    single_match = re.match(r"^\s*///(.*)", lines[i])
                    if single_match:
                        comment_lines.append(single_match.group(1).strip())
                        i += 1
                    else:
                        break
                comment_content = "\n".join(comment_lines)
                entity = find_next_entity(lines, i)
                doxygen_comments.append({"comment": comment_content, "entity": entity})
                continue  # 已经更新了 i 的值

            # 检查 Fortran 单行注释 ! ...，仅匹配行首的 !
            single_line_fortran = re.match(r"^\s*!(.*)", line)
            if single_line_fortran:
                comment_lines = []
                while i < len(lines):
                    single_match = re.match(r"^\s*!(.*)", lines[i])
                    if single_match:
                        comment_lines.append(single_match.group(1).strip())
                        i += 1
                    else:
                        break
                comment_content = "\n".join(comment_lines)
                entity = find_next_entity(lines, i)
                doxygen_comments.append({"comment": comment_content, "entity": entity})
                continue  # 已经更新了 i 的值

            i += 1
    except Exception as e:
        logging.warning(f"无法读取文件 {file_path}，错误: {e}")

    return doxygen_comments


def find_next_entity(lines, start_index):
    """
    从 start_index 开始，查找下一个函数或类的定义。
    返回函数或类的名称，如果未找到则返回 '未知实体'。
    """
    for i in range(start_index, min(start_index + 10, len(lines))):  # 查找接下来的10行
        line = lines[i].strip()
        # 匹配函数定义（简化版）
        func_match = re.match(
            r"^[\w:<>,\s*&]+?\s+(\w+)::(\w+)\s*\(.*\)\s*(const)?\s*{?", line
        )
        if func_match:
            return f"函数 {func_match.group(2)}（类 {func_match.group(1)}）"
        func_match_simple = re.match(
            r"^[\w:<>,\s*&]+?\s+(\w+)\s*\(.*\)\s*(const)?\s*{?", line
        )
        if func_match_simple:
            return f"函数 {func_match_simple.group(1)}"

        # 匹配类定义
        class_match = re.match(r"^class\s+(\w+)", line)
        if class_match:
            return f"类 {class_match.group(1)}"

        # 匹配结构体定义
        struct_match = re.match(r"^struct\s+(\w+)", line)
        if struct_match:
            return f"结构体 {struct_match.group(1)}"

        # 匹配命名空间定义
        namespace_match = re.match(r"^namespace\s+(\w+)", line)
        if namespace_match:
            return f"命名空间 {namespace_match.group(1)}"

    return "未知实体"


def extract_all_doxygen_comments(directory):
    """提取目录中所有 .h, .cpp, .f90 文件的 Doxygen 注释"""
    all_comments = {}
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                file_path = os.path.join(root, file)
                comments = extract_doxygen_comments(file_path)
                if comments:
                    relative_path = os.path.relpath(file_path, directory)
                    all_comments[relative_path] = comments
    return all_comments


def generate_directory_structure(directory):
    """生成当前目录的结构列表"""
    structure = []
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, "").count(os.sep)
        indent = " " * 4 * level
        structure.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = " " * 4 * (level + 1)
        for f in files:
            structure.append(f"{sub_indent}{f}")
    return "\n".join(structure)


def extract_main_cpp_comments(directory):
    """提取 main.cpp 中的 Doxygen 注释"""
    main_cpp_path = os.path.join(directory, "main.cpp")
    if not os.path.exists(main_cpp_path):
        logging.error(f"未找到 main.cpp 文件在 {directory}")
        return ""
    comments = extract_doxygen_comments(main_cpp_path)
    return "\n".join(
        [
            f"注释 {idx} ({item['entity']}):\n{indent_comment(item['comment'])}\n"
            for idx, item in enumerate(comments, 1)
        ]
    )


def extract_main_cpp_source(directory):
    """提取 main.cpp 的源代码"""
    main_cpp_path = os.path.join(directory, "main.cpp")
    if not os.path.exists(main_cpp_path):
        logging.error(f"未找到 main.cpp 文件在 {directory}")
        return ""
    try:
        with open(main_cpp_path, "r", encoding="utf-8") as f:
            source = f.read()
        return source
    except Exception as e:
        logging.warning(f"无法读取 main.cpp 文件，错误: {e}")
        return ""


def copy_to_clipboard(text):
    """将文本复制到剪贴板"""
    try:
        # 使用 pyperclip
        pyperclip.copy(text)
        logging.info("已将内容复制到剪贴板")
    except Exception as e:
        logging.error(f"复制到剪贴板失败，错误: {e}")

    # 如果不使用 pyperclip，请使用 subprocess 调用 clip（Windows）
    """
    try:
        process = subprocess.Popen(['clip'], stdin=subprocess.PIPE, close_fds=True)
        process.communicate(input=text.encode('utf-8'))
        logging.info("已将内容复制到剪贴板")
    except Exception as e:
        logging.error(f"复制到剪贴板失败，错误: {e}")
    """


def format_comments(all_comments):
    """格式化所有文件的 Doxygen 注释"""
    formatted = ""
    for file, comments in all_comments.items():
        formatted += f"文件: {file}\n"
        for idx, item in enumerate(comments, 1):
            entity = item["entity"]
            comment = indent_comment(item["comment"])
            formatted += f"  注释 {idx} ({entity}):\n{comment}\n\n"
    return formatted


def indent_comment(comment, indent_level=4):
    """为注释添加缩进"""
    indent = " " * indent_level
    indented = "\n".join([f"{indent}{line}" for line in comment.split("\n")])
    return indented


# 示例用法
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="轻量级目录结构与Doxygen注释生成器")
    parser.add_argument("startpath", help="项目起始路径")
    args = parser.parse_args()

    # 生成目录结构
    structure = generate_directory_structure(args.startpath)

    # 提取所有支持文件的 Doxygen 注释
    all_comments = extract_all_doxygen_comments(args.startpath)
    formatted_comments = format_comments(all_comments)

    # 提取 main.cpp 的 Doxygen 注释
    main_comments = extract_main_cpp_comments(args.startpath)

    # 提取 main.cpp 的源代码
    main_source = extract_main_cpp_source(args.startpath)

    # 组合输出
    output = (
        f"目录结构:\n{structure}\n\n"
        f"所有文件的 Doxygen 注释:\n{formatted_comments}\n\n"
        f"main.cpp 的 Doxygen 注释:\n{main_comments}\n\n"
        f"main.cpp 的源代码:\n{main_source}"
    )

    # 打印输出（可选）
    print(output)

    # 复制到剪贴板
    copy_to_clipboard(output)
