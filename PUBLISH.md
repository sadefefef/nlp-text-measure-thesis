# 发布说明（GitHub）

本目录是一个完整、可直接推送的 Git 仓库（含全部论文材料、代码、数据与结果）。

## 推送到 GitHub

在具备 GitHub 访问权限的机器上执行：

```bash
cd nlp-book-review-thesis

# 方式一：使用 gh CLI
gh repo create nlp-text-measure-thesis --public --source=. --push

# 方式二：手动
git remote add origin https://github.com/<你的用户名>/nlp-text-measure-thesis.git
git branch -M main
git push -u origin main
```

推送后可在仓库 Settings → Pages 中开启 GitHub Pages（选择 main 分支根目录），
即可获得形如 `https://<用户名>.github.io/nlp-text-measure-thesis/` 的外网访问地址，
与本地 `index.html` 完全一致。

## 说明

- 本仓库在制作环境中已完成 `git init` 与全部提交；
- 若在受控网络环境下（github.com 不可达），可先在本地完成审查，再于具备外网的机器上执行上述推送；
- 论文在线版（本目录 `index.html`）已通过 WorkBuddy「应用发布」部署到外网，见交付说明中的在线访问地址。
