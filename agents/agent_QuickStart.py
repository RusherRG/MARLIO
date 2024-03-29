import sys

sys.path.append("..")
from helpers import model


class Strategy:
    def __init__(self, env, config, logger, logdir):
        self.env = env
        self.config = config

    def act(self, state):
        player_view = state[2]
        for unit in player_view.game.units:
            if unit.player_id == player_view.my_id:
                action = self.get_action(unit, player_view.game)
        return action, action

    def get_action(self, unit, game):
        # Replace this code with your own
        def distance_sqr(a, b):
            return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

        nearest_enemy = min(
            filter(lambda u: u.player_id != unit.player_id, game.units),
            key=lambda u: distance_sqr(u.position, unit.position),
            default=None,
        )

        nearest_weapon = min(
            filter(
                lambda box: isinstance(box.item, model.Item.Weapon), game.loot_boxes
            ),
            key=lambda box: distance_sqr(box.position, unit.position),
            default=None,
        )

        target_pos = unit.position
        if unit.weapon is None and nearest_weapon is not None:
            target_pos = nearest_weapon.position
        elif nearest_enemy is not None:
            target_pos = nearest_enemy.position

        # debug.draw(model.CustomData.Log("Target pos: {}".format(target_pos)))

        aim = model.Vec2Double(0, 0)
        if nearest_enemy is not None:
            aim = model.Vec2Double(
                nearest_enemy.position.x - unit.position.x,
                nearest_enemy.position.y - unit.position.y,
            )
        jump = target_pos.y > unit.position.y
        if (
            target_pos.x > unit.position.x
            and game.level.tiles[int(unit.position.x + 1)][int(unit.position.y)]
            == model.Tile.WALL
        ):
            jump = True
        if (
            target_pos.x < unit.position.x
            and game.level.tiles[int(unit.position.x - 1)][int(unit.position.y)]
            == model.Tile.WALL
        ):
            jump = True
        action = {
            unit.id: {
                "aim_x": aim.x,
                "aim_y": aim.y,
                "jump": jump,
                "jump_down": not jump,
                "plant_mine": False,
                "reload": False,
                "shoot": True,
                "swap_weapon": False,
                "velocity": target_pos.x - unit.position.x,
            }
        }
        return action

    def custom_logic(self, cur_state, action, reward, new_state, done, episode):
        # print("No Custom Logic for random agent")
        return

    def save_model(self, fn):
        return
