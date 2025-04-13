import glob
import os

# 대상 경로 설정 (policy 아래의 모든 하위 디렉터리 포함)
target_path = "model/box-close-v2/AESPA-17-*/reward/ternary-100-aug-high-*/*"
# target_path = "pair/box-close-v2/AESPA-17-*/train/ternary-100-aug-high-*.npz"

# 파일 목록 가져오기
files = glob.glob(target_path, recursive=True)

# 파일 목록 출력
if files:
    print("다음 파일이 삭제될 예정입니다:")
    for file in files:
        if os.path.isfile(file):  # 파일인지 확인
            print(file)
else:
    print("삭제할 파일이 없습니다.")

# 실제 삭제 부분 (주석 처리)
for file in files:
    try:
        if os.path.isfile(file):
            os.remove(file)
            print(f"Deleted: {file}")
    except Exception as e:
        print(f"Error deleting {file}: {e}")

# folders = glob.glob("model/**/AESPA-4-*/policy/**/", recursive=True)
# for folder in sorted(folders, key=len, reverse=True):  # 깊은 폴더부터 삭제
#     try:
#         if os.path.isdir(folder) and not os.listdir(folder):  # 비어 있으면 삭제
#             os.rmdir(folder)
#             print(f"Deleted empty folder: {folder}")
#     except Exception as e:
#         print(f"Error deleting folder {folder}: {e}")

# print("파일 및 폴더 삭제 완료.")

# print("파일 목록 출력 완료.")