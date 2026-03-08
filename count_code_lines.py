from pathlib import Path

# ===== 1. 改成你的项目根目录 =====
PROJECT_ROOT = Path(r"D:\MLandAl\TheFinal\AlloyDesignDemo\软件著作权提交")

# ===== 2. 需要统计的源码扩展名 =====
INCLUDE_EXTS = {".py", ".json", ".yaml", ".yml", ".html", ".css", ".js"}

# ===== 3. 需要排除的目录名 =====
EXCLUDE_DIRS = {
    ".git", "__pycache__", "venv", ".venv", "env",
    "build", "dist", "node_modules", "site-packages",
    ".idea", ".vscode"
}

# ===== 4. 需要排除的文件后缀（数据/模型/结果）=====
EXCLUDE_FILE_EXTS = {
    ".csv", ".xlsx", ".xls", ".pkl", ".joblib", ".png", ".jpg", ".jpeg",
    ".pdf", ".docx", ".pptx", ".txt", ".log"
}

def should_exclude(path: Path) -> bool:
    # 目录命中排除
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True

    # 文件后缀排除
    if path.suffix.lower() in EXCLUDE_FILE_EXTS:
        return True

    return False

def count_file_lines(file_path: Path):
    total_lines = 0
    non_empty_lines = 0

    try:
        with file_path.open("r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                if line.strip():
                    non_empty_lines += 1
    except UnicodeDecodeError:
        try:
            with file_path.open("r", encoding="gbk") as f:
                for line in f:
                    total_lines += 1
                    if line.strip():
                        non_empty_lines += 1
        except Exception as e:
            print(f"跳过文件（无法读取）: {file_path}，原因: {e}")
    except Exception as e:
        print(f"跳过文件: {file_path}，原因: {e}")

    return total_lines, non_empty_lines

def main():
    total_files = 0
    total_physical_lines = 0
    total_non_empty_lines = 0

    print(f"正在统计目录: {PROJECT_ROOT}\n")

    for file_path in PROJECT_ROOT.rglob("*"):
        if not file_path.is_file():
            continue

        if should_exclude(file_path):
            continue

        if file_path.suffix.lower() not in INCLUDE_EXTS:
            continue

        total_lines, non_empty_lines = count_file_lines(file_path)

        total_files += 1
        total_physical_lines += total_lines
        total_non_empty_lines += non_empty_lines

        print(f"{file_path} -> 总行数: {total_lines}, 非空行数: {non_empty_lines}")

    print("\n===== 统计结果 =====")
    print(f"统计文件数: {total_files}")
    print(f"总物理行数: {total_physical_lines}")
    print(f"非空行总数: {total_non_empty_lines}")
    print("\n建议软著填报时优先采用：非空行总数")

if __name__ == "__main__":
    main()