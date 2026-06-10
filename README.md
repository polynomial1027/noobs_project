noobs_project

noobs_project 是一个用于练习 Python 开发和 GitHub 协同合作的入门项目。

这个项目的主要目标不是一开始就完成复杂功能，而是帮助我们熟悉：

* 使用 GitHub 管理代码
* 使用 branch 进行分支开发
* 使用 commit 保存修改记录
* 使用 push 上传代码
* 使用 pull request 合并代码
* 使用 conda 管理 Python 环境
* 练习 Python、机器学习、统计计算、OpenCV、Pygame 和图形界面开发

⸻

1. Project Goals

本项目用于练习以下内容：

1. Python 项目结构搭建
2. Conda 环境管理
3. GitHub 双人协作流程
4. 分支开发与合并
5. Pull Request 使用
6. 简单机器学习与统计计算
7. OpenCV 图像处理
8. Pygame 小游戏开发
9. GUI 图形界面实验
10. Jupyter Notebook 数据分析实验

⸻

2. Project Structure

推荐项目结构如下：

noobs_project/
├── README.md
├── environment.yml
├── requirements.txt
├── .gitignore
├── src/
│   ├── main.py
│   ├── calculator.py
│   ├── george_part.py
│   └── friend_part.py
├── notebooks/
│   └── experiment.ipynb
├── tests/
│   └── test_calculator.py
├── data/
│   └── sample.txt
├── docs/
│   └── experiment_log.md
└── conflict_lab/
    └── shared_note.md

说明：

Path	Description
src/	存放主要 Python 源代码
notebooks/	存放 Jupyter Notebook 实验
tests/	存放测试文件
data/	存放小型示例数据
docs/	存放项目文档和实验记录
conflict_lab/	用于练习 Git merge conflict
environment.yml	Conda 环境配置文件
requirements.txt	Pip 备用依赖列表
.gitignore	设置不需要上传到 GitHub 的文件

⸻

3. Environment Setup

本项目推荐使用 Conda 创建环境。

Create Environment

conda env create -f environment.yml

Activate Environment

conda activate noobs

Update Environment

如果 environment.yml 后续有更新，可以执行：

conda env update -f environment.yml --prune

⸻

4. Run the Main Program

进入项目根目录后执行：

python src/main.py

如果运行成功，会看到主程序输出。

⸻

5. Run Tests

本项目使用 pytest 进行基础测试。

pytest

⸻

6. GitHub Collaboration Workflow

我们不直接在 main 分支上开发。

推荐流程是：

main
├── george-dev
└── friend-dev

main 分支用于保存稳定版本。

每个人在自己的分支上开发，完成后通过 Pull Request 合并到 main。

⸻

7. Create a New Branch

每次开始新功能前，先切回 main 并更新：

git switch main
git pull origin main

然后创建新分支：

git switch -c feature/your-feature-name

例如：

git switch -c feature/main-program

⸻

8. Commit Changes

修改代码后，先查看状态：

git status

添加文件：

git add .

提交修改：

git commit -m "describe what you changed"

例如：

git commit -m "add main program"

⸻

9. Push Branch to GitHub

第一次推送新分支：

git push -u origin feature/your-feature-name

之后同一分支再次推送，只需要：

git push

⸻

10. Pull Request

推送分支后，打开 GitHub 仓库页面：

https://github.com/polynomial1027/noobs_project

点击：

Compare & pull request

确认：

base: main
compare: feature/your-feature-name

然后创建 Pull Request。

检查没有问题后，点击：

Merge pull request

把分支合并到 main。

⸻

11. Sync Local Main After Merge

GitHub 上合并 Pull Request 后，本地 main 不会自动更新。

需要执行：

git switch main
git pull origin main

这样本地 main 才会同步 GitHub 上的最新版本。

⸻

12. Collaboration Rules

为了减少冲突，我们约定：

1. 不直接在 main 上开发。
2. 每个功能使用单独分支。
3. 提交信息要写清楚。
4. 合并前先运行程序和测试。
5. 尽量不要两个人同时修改同一个文件的同一部分。
6. 大文件不要直接上传到 GitHub。
7. Conda 环境以 environment.yml 为准。

⸻

13. Example Commands

完整开发流程示例：

git switch main
git pull origin main
git switch -c feature/main-program
# edit files here
git status
git add .
git commit -m "add main program"
git push -u origin feature/main-program

然后在 GitHub 上创建 Pull Request，并合并到 main。

合并后同步本地：

git switch main
git pull origin main

⸻

14. Current Plan

当前项目计划：

* 创建 Conda 环境
* 添加主程序 src/main.py
* 添加基础测试
* 添加 Jupyter Notebook 示例
* 添加 OpenCV 示例
* 添加 Pygame 示例
* 添加简单机器学习示例
* 练习 Pull Request
* 练习 merge conflict
* 完成一次双人协作开发流程

⸻

15. Notes

这个项目是学习型项目，重点是理解协作流程，而不是一开始就追求复杂功能。

每一次小修改都可以作为一次 GitHub 协作练习。

保存后你可以执行：

git status
git add readme.md
git commit -m "add project README"
git push

如果你想把文件名改成 GitHub 标准的大写：

mv readme.md README.md
git add README.md
git commit -m "rename readme to README"
git push