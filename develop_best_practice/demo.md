### 1. 环境搭建
conda create -n dev python=3.9\
conda activate dev\
conda install -c conda-forge git\
conda install -c conda-forge gitflow\
pip install -U pyscaffold\
pip install pyscaffoldext-markdown\
pip install -U commitizen\
pip install -U pytest\
pip install pytest-cov\
pip install -U sphinx\
pip install sphinx-rtd-theme\
\# visual studio code插件\
pylance、pylint、gitlens、autoDocstring等

### 2. Demo project
\# 创建项目，-n项目名，-p包名，--markdown使用md格式，demo_project项目路径\
putup -n demo_project -p demo --markdown demo_project

![](imgs/dir.png)

\# gitflow初始化，全部选择默认，自动创建分支，完成后自动切换至develop分支\
cd demo_project\
git flow init

![](imgs/gitflow_init.png)

\# commitizen初始化\
\# pyproject.toml -> cz_conventional_commits -> commitizen -> semver -> v$version -> Y -> n -> Enter回车\
cz init

![](imgs/cz_init.png)

\# 提交pyproject.toml的修改\
git add .\
cz commit

![](imgs/commit_toml.png)

\# gitlab创建Project name为demo_project的空项目

![](imgs/gitlab.png)

\# 关联远程并推送\
git remote add origin https://gitlab.genomics.cn/fangzhonghai/demo_project.git

git push --all origin

\# gitlab创建issue要求新增功能

![](imgs/issue1.png)

\# gitflow基于develop分支创建feature分支并自动切换\
git flow feature start factorial

![](imgs/start_factorial.png)

\# skeleton.py中开发新增功能

![](imgs/factorial.png)

\# 提交修改，选择feat，根据提示补充信息(Footer可添加issue编号)

![](imgs/commit_factorial.png)

\# test_skeleton.py编写测试代码

![](imgs/test_factorial.png)

\# pytest自动测试开发的功能\
export PYTHONPATH=\`pwd`/src\
pytest tests/test_skeleton.py

![](imgs/pytest_factorial.png)

\# 提交修改，完成feature分支功能的开发，自动合并至develop分支和删除feature分支\
git add .\
cz commit\
git flow feature finish factorial

![](imgs/finish_factorial.png)

\# 基于develop分支创建release分支发布代码，自动更新changelog和版本，合并至master和develop分支；gitlab close issue\
git push --all origin\
git flow release start v0.1.0\
cz bump\
git flow release finish -n v0.1.0&emsp;&emsp;# -n表示不增加tag，否则会增加release分支名的标签

![](imgs/release_factorial.png)

\# 发现bug，gitlab提issue进行漏洞报告

![](imgs/issue2.png)

\# 确认是bug，自动基于master分支创建hotfix分支进行漏洞修复\
git push --all origin\
git flow hotfix start v0.1.1

![](imgs/start_hotfix.png)

\# skeleton.py修复bug

![](imgs/fix_bug.png)

\# 提交修复bug的代码\
git add .\
cz commit

![](imgs/commit_fix.png)

\# 编写测试代码并测试，测试通过后提交修改，自动更新changelog和版本，完成hotfix分支，自动合并至master和develop分支；gitlab close issue\
python -m demo.skeleton 0\
pytest tests/test_skeleton.py\
git add .\
cz commit\
cz bump\
git flow hotfix finish -n v0.1.1&emsp;&emsp;# -n表示不增加tag，否则会增加hotfix分支名的标签\
git push --all origin

![](imgs/test_fib.png)

![](imgs/pytest_fib.png)

![](imgs/commit_test_bug.png)

\# gitlab提issue改进斐波那契数列功能

![](imgs/issue3.png)

\# 创建feature分支进行功能开发\
git flow feature start fibo_list

![](imgs/start_fibo_list.png)

\# skeleton.py中开发提升功能

![](imgs/fibo_list.png)

\# test_skeleton.py编写测试代码并测试\
pytest tests/test_skeleton.py

![](imgs/test_fibo_list.png)

![](imgs/pytest_fibo_list.png)

\# 提交修改，标注breaking change，Footer添加issue编号，完成feature分支\
git add .\
cz commit\
git flow feature finish fibo_list\
git push --all origin

![](imgs/finish_fibo_list.png)

\# 创建release分支，自动生成changelog和版本，发布代码；gitlab close issue\
git flow release start v1.0.0\
cz bump\
git flow release finish -n v1.0.0\
git push --all origin\
git push origin --tags

![](imgs/release_fibo_list.png)

\# VS code GITLENS查看提交记录

![](imgs/gitlens.png)

\# 生成sphinx_rtd_theme样式文档\
sed -i 's/alabaster/sphinx_rtd_theme/g' docs/conf.py\
make -C docs html

![](imgs/makedocs.png)

![](imgs/http_server.png)

![](imgs/docs.png)