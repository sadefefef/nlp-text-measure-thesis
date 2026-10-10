# 发布说明（GitHub）

本目录是一个**完整、可直接推送的 Git 仓库**：已执行 `git init -b main` 并完成首次提交
（commit `357880d`，30 个文件），包含论文全文、DOCX 成稿、全部源码、语料、结果表与图。

## 一、推送到 GitHub（三步）

在**能访问 GitHub 的终端**里执行：

```bash
cd nlp-book-review-thesis

# 1) 关联远程仓库（先在 GitHub 网页端新建空仓库 nlp-text-measure-thesis，不要勾选 README）
git remote add origin https://github.com/<你的用户名>/nlp-text-measure-thesis.git

# 2) 确认分支名为 main
git branch -M main

# 3) 推送
git push -u origin main
```

若本机装了 `gh` CLI，也可以一条命令完成「建仓 + 推送」：

```bash
gh repo create nlp-text-measure-thesis --public --source=. --push
```

仓库建好后，本目录下的 `push-to-github.sh` 也可直接执行（需先填好用户名）。

## 二、开启外网访问（可选）

推送完成后，在仓库 **Settings → Pages** 中选择 `main` 分支根目录，
即可得到形如 `https://<用户名>.github.io/nlp-text-measure-thesis/` 的地址，
内容与本地 `index.html` 完全一致（全部为相对路径，可直接托管）。

## 三、当前已有的外网访问地址

论文在线版已通过 WorkBuddy「应用发布」上线，无需等待 GitHub Pages：

- 论文在线展示页：`https://intelligent-cloud-mall.app.workbuddy.host/nlp-thesis/`

## 四、说明

- 若在受控网络环境下推送报 `schannel: CRYPT_E_NO_REVOCATION_CHECK`，这是证书吊销检查被拦截导致，
  可用下述方式绕过后重试（仅影响本机 git 进程，不改变系统设置）：

  ```bash
  GIT_SSL_NO_VERIFY=1 git -c http.sslBackend=openssl push -u origin main
  ```

- 仓库不含任何密钥或个人信息；`.wbapp_*.genie`、`.workbuddy/` 等本地运行产物已写入 `.gitignore`。
