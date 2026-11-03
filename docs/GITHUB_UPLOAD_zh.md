# GitHub 上传步骤

本文件夹已整理好，上传前可先检查根目录 FILE_MANIFEST.json 和
CODE_PRESERVATION.json。不要把整个 G:/GPTLEARNING 上传，也不要把外层 ZIP
作为仓库中唯一的项目文件。

## 1 创建空仓库

登录 https://github.com/new，新建仓库，例如 CDS535-Adult-Mushroom。
按课程要求选择 Public 或 Private；Private 仓库需要给老师访问权限。
不要勾选自动创建 README、.gitignore 或 License，因为本目录已有说明文件，
并且没有替你们选择代码许可证。

## 2 使用 Git 推送

三份已训练 Adult 模型分别约 34–36 MiB，超过网页单文件 25 MiB 上传限制，
但所有文件都小于 Git 普通推送的 100 MiB 单文件限制，因此请使用 Git。
安装 Git for Windows：https://git-scm.com/download/win 。

在本文件夹打开 PowerShell，执行：

```powershell
git init
git add .
git status
git commit -m "Add Adult and Mushroom data mining project"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/CDS535-Adult-Mushroom.git
git push -u origin main
```

必须把 remote 地址替换成你的真实仓库地址。首次运行 Git 如提示缺少身份，
设置你自己的姓名及邮箱后重新提交：

```powershell
git config user.name "YOUR_NAME"
git config user.email "YOUR_GITHUB_EMAIL_OR_NOREPLY_EMAIL"
```

按提示通过浏览器登录，不要在代码、文档或命令中写入访问令牌。
本目录没有初始化 Git 或配置远程地址，也没有代你上传。

## 3 后续更新

```powershell
git add .
git status
git commit -m "Update project"
git push
```

## 4 核对

- GitHub 页面能看到 Adult 三个文件、Mushroom 六个文件以及比较脚本。
- README 中的目录、数据和结果都存在，代码不是只存放在 ZIP 内。
- 下载仓库后，先核验文件，再按根目录 README 的命令运行。
- .gitattributes 保留原始字节，避免 Git 自动转换换行而导致哈希变化。
- 作业平台要求的 PDF 和幻灯片仍要按课程渠道提交。
