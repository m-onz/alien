# Installing alien

alien is a set of Pure Data externals plus a small command-line parser. The
build uses `make` and `gcc`, so the setup is mostly the same on every platform:
install Pure Data, install a C compiler, build the project, then copy the
compiled externals into Pd's externals folder.

## What gets installed

Running `make install` creates an `alien` folder inside your Pd externals
directory and copies these files into it:

```text
alien
alien_wrap
alien_join
alien_snap
alien_scale
alien_cue
alien_orchestrate
alien-help.pd
```

The compiled file extension depends on your system:

| System | Extension |
| --- | --- |
| Linux | `.pd_linux` |
| macOS | `.pd_darwin` |
| Windows | `.dll` |

## Download alien

If you use Git:

```bash
git clone https://github.com/m-onz/alien.git
cd alien
```

If you downloaded a ZIP file from GitHub, unzip it and open a terminal in the
unzipped `alien` folder before running the commands below.

## Linux

### 1. Install dependencies

Debian or Ubuntu:

```bash
sudo apt update
sudo apt install build-essential puredata puredata-dev git
```

Fedora:

```bash
sudo dnf install gcc make pure-data pure-data-devel git
```

Arch:

```bash
sudo pacman -S base-devel pd git
```

Package names vary between distributions. If your distribution does not provide
a Pd development package, install Pure Data and make sure the Pd header
`m_pd.h` is available.

### 2. Build and test

```bash
make
make test
```

If the build cannot find `m_pd.h`, tell `make` where the Pd headers are:

```bash
make PD_INCLUDES=-I/usr/include/pd
```

### 3. Install for Pure Data

```bash
make install
```

By default this installs to:

```text
~/Documents/Pd/externals/alien
```

To install somewhere else:

```bash
make install objectsdir="$HOME/.local/lib/pd/extra"
```

Do not use `sudo` for a home-directory install: `sudo` sets `HOME=/var/root`,
so the files end up in root's home instead of yours.

## macOS

### 1. Install dependencies

Install Pure Data from:

```text
https://puredata.info/downloads/pure-data
```

Install Apple's command line build tools:

```bash
xcode-select --install
```

If you do not already have Git, install it through the same prompt or through
Homebrew.

### 2. Build and test

```bash
make
make test
```

The Makefile includes the common Pd application header path. If your Pd app has
a different name or version and the build cannot find `m_pd.h`, pass the header
path manually:

```bash
make PD_INCLUDES="-I/Applications/Pd.app/Contents/Resources/src"
```

### 3. Install for Pure Data

```bash
make install
```

By default this installs to:

```text
~/Documents/Pd/externals/alien
```

To install somewhere else:

```bash
make install objectsdir="$HOME/Library/Pd"
```

Do not use `sudo`: it sets `HOME=/var/root`, so the objects land in
`/var/root/Library/Pd/alien` where Pd (and you) will never see them. A
home-directory install needs no root.

## Windows

The simplest Windows setup is MSYS2, because it provides `gcc`, `make`, and a
Unix-like shell that can run this project's Makefile.

### 1. Install dependencies

1. Install Pure Data from:

```text
https://puredata.info/downloads/pure-data
```

2. Install MSYS2 from:

```text
https://www.msys2.org
```

3. Open the "MSYS2 UCRT64" terminal and install the build tools:

```bash
pacman -Syu
pacman -S --needed mingw-w64-ucrt-x86_64-gcc make git
```

If MSYS2 asks you to close and reopen the terminal after `pacman -Syu`, do that
before running the second command.

### 2. Download alien

In the MSYS2 UCRT64 terminal:

```bash
git clone https://github.com/m-onz/alien.git
cd alien
```

If you downloaded a ZIP instead, use `cd` to enter the unzipped folder.

### 3. Build and test

If Pure Data is installed at `C:\Pd`, run:

```bash
make PD_INCLUDES=-IC:/Pd/src
make test
```

If Pure Data is installed somewhere else, replace `C:/Pd/src` with the folder
that contains `m_pd.h`.

Windows paths with spaces can be awkward in Makefiles. If you run into include
path errors with `C:\Program Files\Pd`, either install Pd at `C:\Pd` or copy the
Pd source headers into a simple path and point `PD_INCLUDES` at that path.

### 4. Install for Pure Data

```bash
make install PD_INCLUDES=-IC:/Pd/src
```

By default this installs to (inside your MSYS2 home):

```text
~/Documents/Pd/externals/alien
```

You can choose a different Pd externals folder with `objectsdir`, for example
the standard Windows location:

```bash
make install PD_INCLUDES=-IC:/Pd/src objectsdir="$APPDATA/Pd"
```

## Loading alien in Pure Data

1. Start Pure Data.
2. Open Pd's preferences and add the parent folder that contains the installed
   `alien` folder to Pd's search path.
3. Create an object named `[alien]`.
4. Open `alien-help.pd` from the installed `alien` folder to confirm the
   external loaded correctly.

If Pd cannot create `[alien]`, check that the installed folder contains the
compiled files for your platform and that Pd's search path points to the parent
folder.

## Useful commands

```bash
make help      # show all build targets
make           # build Pd externals and alien_parser
make test      # run parser tests
make install   # install Pd externals
make clean     # remove local build artifacts
```

## Troubleshooting

### `m_pd.h` was not found

Install the Pure Data development headers, or pass the include directory:

```bash
make PD_INCLUDES=-I/path/to/pd/src
```

### `make` was not found

Install your platform's build tools:

| System | Build tools |
| --- | --- |
| Linux | `build-essential`, `base-devel`, or your distribution equivalent |
| macOS | `xcode-select --install` |
| Windows | MSYS2 UCRT64 with `mingw-w64-ucrt-x86_64-gcc make` |

### Pd still cannot load alien

Make sure the architecture matches your Pd installation. On macOS the Makefile
builds both Intel and Apple Silicon by default. On Windows, build from the MSYS2
environment that matches your Pd version.

Also confirm that Pd is looking in the right place. For example, if alien is
installed at:

```text
~/Documents/Pd/externals/alien
```

then Pd's search path should include:

```text
~/Documents/Pd/externals
```

On Windows, if alien is installed at:

```text
%APPDATA%\Pd\alien
```

then Pd's search path should include:

```text
%APPDATA%\Pd
```
