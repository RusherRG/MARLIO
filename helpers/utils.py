def game_config_json(conf):
    j = {
        "seed": None,
        "options_preset": {
            "Custom": {
                "level": "Simple",
                "properties": {
                    "max_tick_count": 3600,
                    "team_size": 1,
                    "ticks_per_second": 60.0,
                    "updates_per_tick": 100,
                    "loot_box_size": {"x": 0.5, "y": 0.5},
                    "unit_size": {"x": 0.9, "y": 1.8},
                    "unit_max_horizontal_speed": 10.0,
                    "unit_fall_speed": 10.0,
                    "unit_jump_time": 0.55,
                    "unit_jump_speed": 10.0,
                    "jump_pad_jump_time": 0.525,
                    "jump_pad_jump_speed": 20.0,
                    "unit_max_health": 100,
                    "health_pack_health": 50,
                    "weapon_params": {
                        "AssaultRifle": {
                            "magazine_size": 20,
                            "fire_rate": 0.1,
                            "reload_time": 1.0,
                            "min_spread": 0.1,
                            "max_spread": 0.5,
                            "recoil": 0.2,
                            "aim_speed": 1.9,
                            "bullet": {"speed": 50.0, "size": 0.2, "damage": 5},
                            "explosion": None,
                        },
                        "Pistol": {
                            "magazine_size": 8,
                            "fire_rate": 0.4,
                            "reload_time": 1.0,
                            "min_spread": 0.05,
                            "max_spread": 0.5,
                            "recoil": 0.5,
                            "aim_speed": 1.0,
                            "bullet": {"speed": 50.0, "size": 0.2, "damage": 20},
                            "explosion": None,
                        },
                        "RocketLauncher": {
                            "magazine_size": 1,
                            "fire_rate": 1.0,
                            "reload_time": 1.0,
                            "min_spread": 0.1,
                            "max_spread": 0.5,
                            "recoil": 1.0,
                            "aim_speed": 1.0,
                            "bullet": {"speed": 20.0, "size": 0.4, "damage": 30},
                            "explosion": {"radius": 3.0, "damage": 50},
                        },
                    },
                    "mine_size": {"x": 0.5, "y": 0.5},
                    "mine_explosion_params": {"radius": 3.0, "damage": 50},
                    "mine_prepare_time": 1.0,
                    "mine_trigger_time": 0.5,
                    "mine_trigger_radius": 1.0,
                    "kill_score": 1000,
                },
            }
        },
        "players": [
            {
                "Tcp": {
                    "host": None,
                    "port": 31000,
                    "accept_timeout": None,
                    "timeout": None,
                    "token": None,
                }
            },
            "Quickstart",
        ],
    }
    return j
