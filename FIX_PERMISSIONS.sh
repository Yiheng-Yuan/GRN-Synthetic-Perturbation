#!/bin/bash
# 解决 macOS 权限问题并设置 Git 仓库

echo "=========================================="
echo "macOS 权限问题解决方案"
echo "=========================================="

# 问题诊断
echo ""
echo "问题原因："
echo "1. Claude Code 创建的目录有 macOS 扩展属性保护"
echo "2. Git 无法写入 .git/config 和 hooks/"
echo "3. 需要手动在终端修复权限"
echo ""

# 解决方案
echo "=========================================="
echo "解决步骤（请在终端执行）："
echo "=========================================="
echo ""

echo "步骤 1: 打开终端（Terminal.app）"
echo ""

echo "步骤 2: 进入项目目录"
echo "cd /Users/larryheng/Documents/Claude/GRN-Synthetic-Perturbation"
echo ""

echo "步骤 3: 清理扩展属性"
echo "sudo xattr -cr ."
echo "# 会要求输入密码"
echo ""

echo "步骤 4: 修复目录权限"
echo "sudo chmod -R u+w ."
echo "sudo chown -R \$(whoami) ."
echo ""

echo "步骤 5: 删除损坏的 .git 目录"
echo "rm -rf .git"
echo ""

echo "步骤 6: 重新初始化 Git（在用户目录临时操作）"
echo "cd ~"
echo "mkdir -p temp_git_test"
echo "cd temp_git_test"
echo "git init"
echo "cd -"
echo "rm -rf ~/temp_git_test"
echo ""

echo "步骤 7: 在项目目录初始化 Git"
echo "git init"
echo ""

echo "步骤 8: 配置 Git"
echo "git config user.name \"Yiheng Yuan\""
echo "git config user.email \"your-email@example.com\""
echo ""

echo "步骤 9: 添加远程仓库"
echo "git remote add origin https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation.git"
echo ""

echo "步骤 10: 创建 .gitignore"
cat << 'GITIGNORE' > /dev/stdout
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/
.venv/
venv/

# IDE
.vscode/
.idea/
*.swp

# Testing
.pytest_cache/
.coverage
htmlcov/

# Data (不要提交实验数据)
data/
results/
checkpoints/
*.npy
*.h5

# Logs
*.log

# OS
.DS_Store
EOF
GITIGNORE
echo ""

echo "步骤 11: 添加并提交文件"
echo "git add ."
echo "git status"
echo ""

echo "步骤 12: 创建 dev 分支并提交"
echo "git checkout -b dev"
cat << 'COMMITMSG' > /dev/stdout
git commit -m "feat: 完善库实现 v0.2.0

核心新增功能：
- 实现问题二（参数辨识）完整功能 (parameter_sets.py)
- 添加配置管理系统 (config.py)
- 实现效率学习机制 (efficiency.py)
- 添加完整集成测试 (test_integration.py)
- 提供生产级示例脚本 (run_complete_experiment.py)

改进：
- 解决硬编码问题
- 增强信息隔离设计
- 完善文档和类型注解

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
COMMITMSG
echo ""

echo "步骤 13: 创建 experiment 分支"
echo "git branch experiment"
echo ""

echo "步骤 14: 推送到 GitHub"
echo "git push -u origin dev"
echo "git push -u origin experiment"
echo ""

echo "=========================================="
echo "快速命令（复制粘贴到终端）："
echo "=========================================="
echo ""
cat << 'QUICKCMD'
cd /Users/larryheng/Documents/Claude/GRN-Synthetic-Perturbation
sudo xattr -cr .
sudo chmod -R u+w .
sudo chown -R $(whoami) .
rm -rf .git
git init
git config user.name "Yiheng Yuan"
git config user.email "your-email@example.com"
git remote add origin https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation.git

cat > .gitignore << 'EOF'
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
data/
results/
checkpoints/
*.log
.DS_Store
EOF

git add .
git checkout -b dev
git commit -m "feat: 完善库实现 v0.2.0

核心新增功能：
- 实现问题二（参数辨识）
- 添加配置管理系统
- 实现效率学习机制
- 添加完整集成测试

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"

git branch experiment
git push -u origin dev
git push -u origin experiment
QUICKCMD

echo ""
echo "=========================================="
echo "注意事项："
echo "=========================================="
echo "1. 必须在终端（Terminal）中执行，不能在 Claude Code 中"
echo "2. sudo 命令需要输入密码"
echo "3. 记得替换 email 为你的真实邮箱"
echo "4. 首次 push 可能需要 GitHub 认证"
echo ""
