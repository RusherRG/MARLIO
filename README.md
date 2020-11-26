# MARLIO
Training agents to compete and/or cooperate with other agents in an already existing 2D environment using Deep Reinforcement Learning algorithms to win games and defeat other agents.

## Setup

### Game server

```shell
cd app-src/
sudo apt-get install libsdl2-dev
```

To build, you'll need to install [Rust](https://rustup.rs/).

```shell
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

To run native build:

```shell
cargo run --release
```

To build web version, first install [`cargo-web`](https://github.com/koute/cargo-web):

```shell
cargo install cargo-web
```

Then build, start local server and open browser:

```shell
cargo web start --release --open
```

### Install OpenAI gym

```shell
pip3 install -r requirements.txt
pip3 install -e gym_codeside
```

### Run game server

```shell
cp app-src/target/release/aicup2019 bin/aicup2019
cd bin/
./aicup2019 --config config_test.json
```

### Clients

```shell
cd agent && python3 test.py
```
