# README #

This README would normally document whatever steps are necessary to get your application up and running.

### Setup ###

* Install git lfs (git large file system)
```
$ brew install git-lfs
$ cd your_repo_dir
$ git lfs install
```

* Add the eviroment variable to specify the root loocation of the repository to use the lelative paths in python and blender.
```
$ cd ~
$ nano .bashrc
$ nano .zshrc
```
if you use **bash** or **zsh** and add the line
```
export ML_LIGHT_DIR = $HOME/path_to_repo/ml_light/
```
### Running ###

* Run blender form Terminal to be able read outputs and it load environment variable
```
$ cd /Applications/Blender.app/Contents/MacOS
$ ./blender
```


[Learn Markdown](https://bitbucket.org/tutorials/markdowndemo)


