#!/bin/bash
# Git 仓库设置和提交脚本
# 请在终端手动运行此脚本

set -e

echo "====================================="
echo "GRN-Synthetic-Perturbation Git 设置"
echo "====================================="

# 检查是否在项目目录
if [ ! -d "src/grn_experiment" ]; then
    echo "错误：请在项目根目录运行此脚本"
    exit 1
fi

# 1. 配置 Git 远程仓库
echo ""
echo "步骤 1: 配置远程仓库..."
git remote add origin https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation.git 2>/dev/null || true
git remote set-url origin https://github.com/Yiheng-Yuan/GRN-Synthetic-Perturbation.git

# 2. 创建 .gitignore
echo ""
echo "步骤 2: 创建 .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Data and results (DO NOT commit experimental data)
data/
results/
checkpoints/
*.npy
*.npz
*.h5
*.pkl

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# Frozen configs (commit these)
!frozen_config.json
EOF

# 3. 添加所有新文件
echo ""
echo "步骤 3: 添加新文件到 git..."
git add .

# 4. 查看状态
echo ""
echo "步骤 4: 当前状态..."
git status

# 5. 提交到 dev 分支
echo ""
echo "步骤 5: 提交改进..."
read -p "请输入你的 Git 用户名: " git_user
read -p "请输入你的 Git 邮箱: " git_email

git config user.name "$git_user"
git config user.email "$git_email"

# 确保在 dev 分支
git checkout -b dev 2>/dev/null || git checkout dev

# 提交
git commit -m "feat: 完善库实现 v0.2.0

核心新增功能：
- 实现问题二（参数辨识）完整功能 (parameter_sets.py)
- 添加配置管理系统 (config.py)
- 实现效率学习机制 (efficiency.py)
- 添加完整集成测试 (test_integration.py)
- 提供生产级示例脚本 (run_complete_experiment.py)

改进：
- 解决硬编码问题，所有阈值通过配置管理
- 增强信息隔离设计
- 完善文档和类型注解
- 统一模块导出接口 (__init__.py)

详细变更：
1. parameter_sets.py - 相容参数集构建、收缩指标、参数导向选样
2. config.py - 类型安全的配置冻结和验证
3. efficiency.py - 贝叶斯效率估计替代固定先验
4. test_integration.py - 12轮完整循环测试
5. run_complete_experiment.py - 端到端实验流程
6. quickstart_notebook.py - 交互式快速入门
7. README_v2.md - 更新完整文档

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"

echo ""
echo "✅ 提交完成！"
echo ""

# 6. 创建其他分支
echo "步骤 6: 创建分支结构..."
git branch experiment 2>/dev/null || true

echo ""
echo "分支结构已创建："
git branch -a

# 7. 推送指南
echo ""
echo "====================================="
echo "下一步操作："
echo "====================================="
echo ""
echo "1. 推送 dev 分支到 GitHub:"
echo "   git push -u origin dev"
echo ""
echo "2. 推送 experiment 分支:"
echo "   git push -u origin experiment"
echo ""
echo "3. 在 GitHub 上创建 Pull Request:"
echo "   从 dev → main (当代码经过验证后)"
echo ""
echo "4. 日常开发工作流："
echo "   - 在 dev 分支工作: git checkout dev"
echo "   - 实验新功能: git checkout experiment"
echo "   - 验证通过后合并到 main"
echo ""
echo "====================================="
