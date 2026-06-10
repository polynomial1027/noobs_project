# noobs_project

noobs_project 是一个用于练习 Python 开发、Conda 环境管理和 GitHub 协同合作的入门项目。

本项目的重点是熟悉标准协作流程：

text 下载项目 → 创建环境 → 创建任务分支 → 修改代码 → 提交修改 → 推送分支 → 创建 Pull Request → 合并到 main 

---

## 1. Clone Project

第一次使用项目时，先把 GitHub 仓库下载到本地。

```bash 
cd ~/（你要存的文件夹地址） git clone https://github.com/polynomial1027/noobs_project.git
cd noobs_project 
```

如果你已经下载过项目，以后直接进入项目目录即可：

```bash 
cd ~/（你要存的文件夹地址）/noobs_project 
```
---

## 2. Create Conda Environment

本项目使用 Conda 管理 Python 环境。

第一次使用时，执行：

```bash 
conda env create -f environment.yml 
```

创建完成后，激活环境：

```bash 
conda activate noobs 
```

检查 Python 是否可用：
```bash 
python --version 
```

创建并激活 Conda 环境后，可以运行下面的环境检测脚本：

```bash 
python tests/environment_test.py
```

如果所有库都安装成功，会看到类似输出：
```
Environment check passed.
```
---

## 3. Update Conda Environment

如果别人更新了 environment.yml，你需要更新本地环境：
```bash 
conda activate noobs conda env update -f environment.yml --prune 
```
---

## 4. Run Project

进入项目根目录后运行：

```bash 
python src/main.py 
```

如果项目包含测试文件，可以运行：

```bash 
pytest 
```
---

## 5. Branch Rule

不要直接在 main 分支上开发。

main 分支只保存稳定版本。

每次做新任务，都从最新的 main 创建一个新分支。

分支名按照任务命名，不按照人名命名。

推荐格式：

```text 
feature/功能名 
fix/问题名 
docs/文档名 
experiment/实验名 
```
示例：

```text 
feature/main-program 
feature/pygame-demo 
feature/opencv-camera 
fix/test-error 
docs/update-readme 
experiment/sklearn-demo 
```
---

## 6. Daily Start Workflow

每次开始工作前，先进入项目目录：

```bash 
cd ~/（你要存的文件夹地址）/noobs_project 
```

激活 Conda 环境：

```bash 
conda activate noobs 
```

切换到 main：

```bash 
git switch main 
```

拉取 GitHub 上最新的 main：

```bash 
git pull origin main 
```

检查当前状态：

```bash 
git status 
```

如果看到：

```text 
nothing to commit, working tree clean 
```

说明当前工作区是干净的，可以开始新任务。

---

## 7. Create a Task Branch

每次做一个任务，都创建一个新的任务分支。

例如：添加主程序。

```bash 
git switch -c feature/main-program 
```

例如：修改 README。

```bash 
git switch -c docs/update-readme 
```

例如：修复测试错误。

```bash 
git switch -c fix/test-error 
```

例如：添加 Pygame 示例。

```bash 
git switch -c feature/pygame-demo 
```

查看当前分支：

```bash 
git branch 
```

星号 * 所在的分支就是当前分支。

---

## 8. Edit Files

在当前任务分支上修改代码或文档。

例如创建 src/main.py：

```bash 
mkdir -p src  
cat > src/main.py <<'EOF' 
def main():     
    print("Welcome to noobs_project!")     
    print("This is the main program.")     
    print("We are practicing GitHub collaboration.")   
if __name__ == "__main__":     
    main() 
EOF 
```
运行检查：

```bash 
python src/main.py 
```

---

## 9. Check Changes

查看当前修改状态：

```bash 
git status 
```

查看具体修改内容：

```bash 
git diff 
```

查看当前分支：

```bash 
git branch 
```

---

## 10. Commit Changes

添加所有修改：

```bash 
git add . 
```

提交修改：

```bash 
git commit -m "add main program" 
```

提交信息要简短清楚。

推荐示例：

```bash 
git commit -m "add main program" 
git commit -m "update README usage guide" 
git commit -m "fix calculator test" 
git commit -m "add pygame demo" 
git commit -m "add opencv camera example" 
```
---

## 11. Push Branch to GitHub

第一次推送新分支时，需要写完整命令。

例如当前分支是 feature/main-program：

```bash 
git push -u origin feature/main-program 
```

如果当前分支是 docs/update-readme：

```bash 
git push -u origin docs/update-readme 
```

如果当前分支是 fix/test-error：

```bash 
git push -u origin fix/test-error 
```

同一个分支之后再次推送，只需要：

```bash 
git push 
```

---

## 12. Create Pull Request

推送分支后，打开 GitHub 仓库：

```text 
https://github.com/polynomial1027/noobs_project 
```

点击：

```text 
Compare & pull request 
```

确认合并方向：

```text 
base: main compare: your-task-branch 
```

例如：

```text 
base: main compare: feature/main-program 
```

然后点击：

```text 
Create pull request 
```

Pull Request 标题建议写清楚，例如：

```text 
Add main program 
Update README usage guide 
Fix calculator test 
Add pygame demo 
Add opencv camera example 
```

---

## 13. Merge Pull Request

如果 GitHub 显示没有冲突，可以点击：

```text 
Merge pull request 
```

然后点击：

```text 
Confirm merge 
```

这样任务分支的内容就合并进 main 了。

合并后可以删除 GitHub 上的任务分支。

---

## 14. Sync Local Main After Merge

GitHub 上合并 Pull Request 后，本地 main 不会自动更新。

需要回到终端执行：

```bash 
git switch main git pull origin main 
```

确认最新内容已经同步：

```bash 
git log --oneline 
```

运行项目：

```bash 
python src/main.py 
```
---

## 15. Full Workflow Example

下面是一套完整流程，可以直接复制使用。

### Step 1: Start from latest main

```bash 
cd ~/（你要存的文件夹地址）/noobs_project 
conda activate noobs 
git switch main 
git pull origin main 
git status 
```
### Step 2: Create a task branch

```bash 
git switch -c feature/main-program 
```

### Step 3: Edit files

```bash 
mkdir -p src  
cat > src/main.py <<'EOF' 
def main():     
    print("Welcome to noobs_project!")     
    print("This is the main program.")     
    print("We are practicing GitHub collaboration.")   
if __name__ == "__main__":     
    main() EOF 
```

### Step 4: Test locally

```bash 
python src/main.py pytest 
```

### Step 5: Commit changes

```bash 
git status 
git add . 
git commit -m "add main program" 
```

### Step 6: Push branch

```bash 
git push -u origin feature/main-program 
```

### Step 7: Open Pull Request on GitHub

Open:

```text 
https://github.com/polynomial1027/noobs_project 
```

Then click:

```text 
Compare & pull request Create pull request 
```

### Step 8: Merge Pull Request

If there is no conflict, click:

```text 
Merge pull request Confirm merge 
```

### Step 9: Sync local main

```bash 
git switch main 
git pull origin main 
python src/main.py
``` 

---


## 16. Useful Git Commands

查看当前分支：

```bash 
git branch 
```

查看当前状态：

```bash 
git status 
```

查看修改内容：

```bash 
git diff 
```

查看提交历史：

```bash 
git log --oneline 
```

切换到 main：

```bash 
git switch main 
```

拉取最新 main：

```bash 
git pull origin main 
```

查看远程仓库地址：

```bash 
git remote -v 
```

---

## 17. Important Rules

1. 不要直接在 main 分支上开发。
2. 每次任务都从最新的 main 创建新分支。
3. 分支名按任务命名，不按人名命名。
4. 修改前先 git pull origin main。
5. 提交前先 git status。
6. 合并前先本地运行程序。
7. 合并 Pull Request 后，本地要重新 git pull origin main。
8. Conda 环境以 environment.yml 为准。
9. 大文件不要上传到 GitHub。
10. 如果不确定当前在哪个分支，先运行 git branch。
