import glob
import os
import shutil

# 검색 패턴 설정
search_pattern = "dataset/*/CUDA*"
# search_pattern = "model/*/*/reward/ternary-500-aug-10000-bucket-knn-ratio-10-20-uncert-2.3**"
paths = glob.glob(search_pattern, recursive=True)

# 필터링: 실제 존재하는 경로만
paths = [p for p in paths if os.path.exists(p)]

# 경로를 길이 내림차순으로 정렬 (파일 → 폴더 순으로 삭제)
paths.sort(key=lambda x: -len(x))

if not paths:
    print("삭제할 항목이 없습니다.")
else:
    print("다음 항목을 삭제합니다:")
    for path in paths:
        if os.path.isfile(path):
            print(f"[FILE ] {path}")
            os.remove(path)
        elif os.path.isdir(path):
            print(f"[DIR  ] {path}")
            # shutil.rmtree(path)