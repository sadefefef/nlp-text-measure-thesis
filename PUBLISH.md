# 发布说明

## 一、已发布地址

| 用途 | 地址 |
| --- | --- |
| GitHub 仓库 | <https://github.com/sadefefef/nlp-text-measure-thesis> |
| 论文在线展示页（已上线） | <https://intelligent-cloud-mall.app.workbuddy.host/nlp-thesis/> |

本仓库含两个分支：

- **`main`** —— 完整论文材料（32 个文件：论文 DOCX 成稿、Markdown 全文、源码、语料、9 张结果表、6 张插图、在线页 `index.html`）
- **`gh-pages`** —— 与 `main` 相同的内容，作为 GitHub Pages 的发布分支

## 二、开启 GitHub Pages（一步，可选）

仓库 → **Settings → Pages** → Source 选 **Deploy from a branch** → Branch 选 **`gh-pages`** → `/ (root)` → Save。
约 1 分钟后即可通过 <https://sadefefef.github.io/nlp-text-measure-thesis/> 访问
（`index.html` 全部使用相对路径，可直接托管）。

## 三、后续更新内容

```bash
git add -A
git commit -m "你的修改说明"
git push origin main
```

## 四、本机推送注意事项

- 若在受控网络（带 TLS 代理）下报 `schannel: CRYPT_E_NO_REVOCATION_CHECK` 或
  `OpenSSL ... unable to get local issuer certificate`，用下面的方式推送：

  ```bash
  GIT_SSL_NO_VERIFY=1 git -c http.sslBackend=openssl push origin main
  ```

- 该网络下推送偶发 `OpenSSL SSL_read: unexpected eof while reading`（代理抖动），
  **直接重试 1–2 次即可成功**。
- 在正常网络环境下，直接 `git push` 即可，无需上述参数。
- 仓库不含任何密钥或个人信息；`.wbapp_*.genie`、`.workbuddy/` 等本地运行产物已写入 `.gitignore`。
