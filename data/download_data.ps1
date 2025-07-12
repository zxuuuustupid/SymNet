# 创建临时目录
New-Item -ItemType Directory -Force -Path tmp

# 下载数据集
Invoke-WebRequest -Uri "https://www.cs.utexas.edu/~tushar/attribute-ops/attr-ops-data.tar.gz" -OutFile "tmp\attr-ops-data.tar.gz"
Invoke-WebRequest -Uri "http://wednesday.csail.mit.edu/joseph_result/state_and_transformation/release_dataset.zip" -OutFile "tmp\mitstates.zip"
Invoke-WebRequest -Uri "http://vision.cs.utexas.edu/projects/finegrained/utzap50k/ut-zap50k-images.zip" -OutFile "tmp\utzap.zip"
Invoke-WebRequest -Uri "http://www.cs.cmu.edu/~spurushw/publication/compositional/compositional_split_natural.tar.gz" -OutFile "tmp\natural.tar.gz"

Write-Output "Data downloaded. Extracting files..."

# 解压 attr-ops-data.tar.gz（需要安装 tar）
tar -zxvf "tmp\attr-ops-data.tar.gz" --strip-components=1

# 重命名原始数据文件夹
Rename-Item -Path "data\mit-states" -NewName "mit-states-original"
Rename-Item -Path "data\ut-zap50k" -NewName "ut-zap50k-original"

# 删除不需要的文件夹
Remove-Item -Recurse -Force cv, tensor-completion, data

# 解压 MIT-States 图片部分
Expand-Archive -Path "tmp\mitstates.zip" -DestinationPath "mit-states-original"
Move-Item "mit-states-original\release_dataset\images" "mit-states-original\images"
Remove-Item -Recurse -Force "mit-states-original\release_dataset"

# 替换 MIT-States 图片文件夹中空格为下划线（需 PowerShell 5.0+）
Get-ChildItem "mit-states-original\images" | Rename-Item -NewName { $_.Name -replace " ", "_" }

# 解压 UT-Zappos50k 并重组结构
Expand-Archive -Path "tmp\utzap.zip" -DestinationPath "ut-zap50k-original"
Move-Item "ut-zap50k-original\ut-zap50k-images" "ut-zap50k-original\_images"

# 运行 Python 脚本重新组织 zap 图片（需你本地有该脚本）
python reorganize_utzap.py

# 删除临时 _images 文件夹
Remove-Item -Recurse -Force "ut-zap50k-original\_images"

# 解压自然划分 split
tar -zxvf "tmp\natural.tar.gz"

# 移动 metadata 和 split 文件夹
Move-Item "mit-states\metadata_compositional-split-natural.t7" "mit-states\metadata.t7"
Move-Item "ut-zap50k\metadata_compositional-split-natural.t7" "ut-zap50k\metadata.t7"
Move-Item "mit-states\compositional-split-natural" "mit-states\compositional-split"
Move-Item "ut-zap50k\compositional-split-natural" "ut-zap50k\compositional-split"

# 重命名为带 natural 后缀的版本
Rename-Item -Path "mit-states" -NewName "mit-states-natural"
Rename-Item -Path "ut-zap50k" -NewName "ut-zap50k-natural"

# 创建软连接（符号链接）指向图片目录（需以管理员身份运行 PowerShell）
cmd /c mklink /D "mit-states-natural\images" "..\mit-states-original\images"
cmd /c mklink /D "ut-zap50k-natural\images" "..\ut-zap50k-original\images"

# （可选）清除临时文件
# Remove-Item -Recurse -Force tmp
