import model
import random
from copy import copy


class MyStrategy:
    def __init__(self):
        self.tick = 0
        self.firetick = 5
        self.fire = 0
        self.allow_fire = False
        self.prev = model.Vec2Float(0, 0)
        self.direction = 1
        self.swap = False
        self.spr = 1
        self.dist = 0
        self.override = False
        self.target_pos = None
        self.ttick = 0

    def get_action(self, unit, game, debug):
        # Replace this code with your own
        def distance_sqr(a, b):
            return (a.x - b.x) ** 2 + (a.y - b.y) ** 2

        def si(a):
            if a > 0:
                return 1
            elif a < 0:
                return - 1
            else:
                return 0

        def sign(a, b, Y, X):
            # debug.draw(model.CustomData.Line(model.Vec2Float(a.x, a.y), model.Vec2Float(b.x, b.y), 0.1, model.ColorFloat(1, 0, 0, 1)))
            # debug.draw(model.CustomData.Line(model.Vec2Float(a.x, a.y), model.Vec2Float(
            #     X, Y), 0.1, model.ColorFloat(1, 0, 0, 1)))
            # debug.draw(model.CustomData.Line(model.Vec2Float(b.x, b.y), model.Vec2Float(
            #     X, Y), 0.1, model.ColorFloat(1, 0, 0, 1)))

            return si((b.x - a.x) * (Y - a.y) - (b.y - a.y) * (X - a.x))

        def check_los(a, b, map):
            for mx in range(len(map)):
                for my in range(len(map[mx])):
                    if (a.x < b.x and (mx > a.x and mx < b.x)) or (a.x >= b.x and (mx <= a.x and mx >= b.x)):
                        if (a.y < b.y and (my > a.y and my < b.y)) or (a.y >= b.y and (my <= a.y and my >= b.y)):
                            if map[mx][my] == model.Tile.WALL:
                                if sign(a, b, my, mx) == sign(a, b, my + 1, mx) == sign(a, b, my + 1, mx + 1) == sign(a, b, my, mx + 1):
                                    continue
                                else:
                                    return False
            return True

        def los(a, b, map):
            return check_los(model.Vec2Double(a.x, a.y+1.5), model.Vec2Double(b.x, b.y + 1 + self.spr), map) \
                and check_los(model.Vec2Double(a.x, a.y+1), model.Vec2Double(b.x, b.y + 1), map) \
                and check_los(model.Vec2Double(a.x, a.y+0.5), model.Vec2Double(b.x, b.y + 1 - self.spr), map)

        def jump_calc(target_pos, unit, game):
            jump = (target_pos.y + 0.5) > unit.position.y
            if abs(target_pos.y - unit.position.y) <= 0.3:
                jump = False
            if target_pos.x > unit.position.x and abs(target_pos.x - unit.position.x) > 0.1 and game.level.tiles[int(unit.position.x + 1)][int(unit.position.y)] == model.Tile.WALL:
                jump = True
            if target_pos.x < unit.position.x and abs(target_pos.x - unit.position.x) > 0.1 and game.level.tiles[int(unit.position.x - 1)][int(unit.position.y)] == model.Tile.WALL:
                jump = True

            return jump

        nearest_enemy = min(
            filter(lambda u: u.player_id != unit.player_id, game.units),
            key=lambda u: distance_sqr(u.position, unit.position),
            default=None)

        nearest_weapon = min(
            filter(lambda box: isinstance(
                box.item, model.Item.Weapon), game.loot_boxes),
            key=lambda box: distance_sqr(box.position, unit.position),
            default=None)

        nearest_health = min(
            filter(lambda box: isinstance(
                box.item, model.Item.HealthPack), game.loot_boxes),
            key=lambda box: distance_sqr(box.position, unit.position),
            default=None)

        target_pos = copy(unit.position)
        dist = distance_sqr(nearest_enemy.position, unit.position)
        override = copy(self.override)

        swap = False
        if nearest_weapon is not None:
            if unit.weapon is None:
                target_pos = copy(nearest_weapon.position)
            elif unit.weapon is not None and (unit.weapon.typ < nearest_weapon.item.weapon_type) and distance_sqr(unit.position, nearest_weapon.position) < 64:
                target_pos = copy(nearest_weapon.position)
                swap = True
            # elif nearest_enemy is not None:
            #     target_pos = nearest_enemy.position  # to change heavily

        jump = jump_calc(target_pos, unit, game)

        aim = model.Vec2Double(0, 0)
        if nearest_enemy is not None:
            aim = model.Vec2Double(
                nearest_enemy.position.x - unit.position.x,
                nearest_enemy.position.y - unit.position.y)

        if unit.weapon is not None:
            los_ans = los(unit.position, nearest_enemy.position,
                          game.level.tiles)
            # debug.draw(model.CustomData.Log("LOS: {}".format(los_ans)))
            if dist < 60:
                target_pos.y -= 10 * \
                    (nearest_enemy.position.y - unit.position.y)
                target_pos.x -= 10 * \
                    (nearest_enemy.position.x - unit.position.x)
                override = True
            if los_ans and self.fire < self.firetick:
                self.fire += 1
                target_pos = nearest_enemy.position
            if self.fire > self.firetick/2 and los_ans:
                self.allow_fire = True
                if (distance_sqr(self.prev, unit.position) > dist):
                    if dist < 60 and target_pos.x == unit.position.x:
                        target_pos.x -= 10 * \
                            (nearest_enemy.position.x - unit.position.x)
                else:
                    target_pos = copy(unit.position)
                jump = False
                if dist < 60:
                    if game.level.tiles[int(unit.position.x)][int(unit.position.y - 1)] == model.Tile.WALL and game.level.tiles[int(unit.position.x - 1 * self.direction)][int(unit.position.y)] == model.Tile.WALL:
                        jump = True
                    elif nearest_enemy.position.y < unit.position.y:
                        jump = True
            elif self.fire > -1 * (self.firetick) and los_ans == False:
                self.fire -= 1
                self.allow_fire = False
            if los_ans == False:
                self.allow_fire = False
                if self.direction*nearest_enemy.position.x < self.direction*self.prev.x and target_pos.x == unit.position.x:
                    target_pos = copy(unit.position)
                else:
                    target_pos = nearest_enemy.position
            # Healer
            if nearest_health is not None:
                x = game.properties.unit_max_health - unit.health
                if x >= game.properties.health_pack_health \
                        or ((unit.health > nearest_enemy.health) and distance_sqr(nearest_enemy.position, nearest_health.position) < 4) \
                        or x <= game.properties.unit_max_health/2:
                    y = 0
                else:
                    j = si(nearest_enemy.position.x -
                           nearest_health.position.x)
                    if game.level.tiles[int(nearest_health.position.x + j)][int(nearest_health.position.y)] == model.Tile.WALL or \
                            game.level.tiles[int(nearest_health.position.x + j)][int(nearest_health.position.y - 1)] == model.Tile.EMPTY or \
                            game.level.tiles[int(nearest_health.position.x + j)][int(nearest_health.position.y - 1)] == model.Tile.JUMP_PAD:
                        y = j * -1
                    else:
                        y = j
                if (x == 0) or (target_pos.x == nearest_weapon.position.x and distance_sqr(unit.position, nearest_weapon.position) < distance_sqr(unit.position, nearest_health.position)):
                    vel = si(target_pos.x - unit.position.x)
                else:
                    if dist > 9:
                        target_pos.x = nearest_health.position.x + y
                        target_pos.y = nearest_health.position.y
                        vel = si(target_pos.x - unit.position.x)

        # End of Healing

        # Matrix Code
        def collision(pos, bullet):
            a = copy(bullet.position)
            b = copy(bullet.position)
            b.x += bullet.velocity.x
            b.y += bullet.velocity.y
            size = unit.size
            X = pos.x - (size.x / 2)
            Y = pos.y
            if sign(a, b, Y, X) == sign(a, b, Y, X + size.x) == sign(a, b, Y + size.y, X) == sign(a, b, Y + size.y, X + size.x):
                return False
            if si(bullet.position.x - pos.x) == si(bullet.velocity.x) and si(bullet.position.y - pos.y) == si(bullet.velocity.y):
                return False
            return True

        def draw_bullet_path(bullet):
            pos = bullet.position
            vel = bullet.velocity
            # debug.draw(model.CustomData.Line(model.Vec2Float(pos.x, pos.y),
            #             model.Vec2Float(pos.x + vel.x, pos.y + vel.y), 0.1, model.ColorFloat(1, 0, 0, 1)))

        def get_line(a, b, type='none'):
            line = lambda **kwargs: type("Object", (), kwargs)()
            line.x1 = a.x
            line.y1 = a.y
            line.x2 = b.x
            line.y2 = b.y
            line.a = b.y - a.y
            line.b = a.x - b.x
            line.c = line.a * a.x + line.b * a.y
            line.type = type
            return line

        def intersection(l1, l2):
            det = l1.a * l2.b - l2.a * l1.b

            if det == 0:
                return False
            x = (l2.b * l1.c - l1.b * l2.c) / det
            y = (l1.a * l2.c - l2.a * l1.c) / det

            if min(int(l1.x1), int(l1.x2)) <= int(x) <= max(int(l1.x1), int(l1.x2)) \
                    and min(int(l1.y1), int(l1.y2)) <= int(y) <= max(int(l1.y1), int(l1.y2)):
                return [model.Vec2Float(x, y), l1.type]
            else:
                return False

        def maxpoint(bullet):
            pos = bullet.position
            velo = bullet.velocity
            bullet_line = get_line(model.Vec2Float(
                pos.x, pos.y), model.Vec2Float(pos.x + velo.x, pos.y + velo.y))
            size = unit.size
            X = unit.position.x - (size.x / 2)
            Y = unit.position.y
            u1 = get_line(model.Vec2Float(X, Y),
                          model.Vec2Float(X + size.x, Y),
                          'bottom')
            u2 = get_line(model.Vec2Float(X, Y),
                          model.Vec2Float(X, Y + size.y),
                          'left')
            u3 = get_line(model.Vec2Float(X + size.x, Y),
                          model.Vec2Float(X + size.x, Y + size.y),
                          'right')
            u4 = get_line(model.Vec2Float(X, Y + size.y),
                          model.Vec2Float(X + size.x, Y + size.y),
                          'top')
            u = [u1, u2, u3, u4]
            j = []
            for i in u:
                a = intersection(i, bullet_line)
                if a != False:
                    j.append(a)
            return min(j, key=lambda l: distance_sqr(l[0], bullet.position))

        def avoid_bullets(bullets):
            # will just decide target position
            nb = min(bullets,
                     key=lambda b: distance_sqr(b.position, unit.position),
                     default=None)

            dbx = abs(nb.position.x - unit.position.x)
            bx = abs(nb.velocity.x)
            by = abs(nb.velocity.y)
            jump_time = game.properties.unit_jump_time
            int_pt = maxpoint(nb)
            h = abs(int_pt[0].y - unit.position.y)
            time_remaning_sqr = distance_sqr(
                int_pt[0], nb.position) / (bx ** 2 + by ** 2)
            time_needed = h / game.properties.unit_jump_speed
            mid = copy(unit.position)
            mid.y += unit.size.y / 2
            if time_remaning_sqr <= (time_needed**2)+0.1:
                if int_pt[1] == 'left' or int_pt[1] == 'right':
                    if int_pt[0].y > mid.y and game.level.tiles[int(unit.position.x)][int(unit.position.y - 1)] != model.Tile.WALL:
                        targ.y = unit.position.y - 2
                    else:
                        targ.y = unit.position.y + 2
                    targ.x = copy(unit.position.x)
                else:
                    targ.x = unit.position.x + si(mid.x - int_pt[0].x)*2
                    targ.y = copy(unit.position.y)
                # debug.draw(model.CustomData.Log("Targ: {}".format(targ)))
                return True
            return False

        def get_away(bullet, map):
            a = bullet.position
            b = copy(bullet.position)
            b.x += (bullet.velocity.x*3)
            b.y += (bullet.velocity.y*3)
            w = []
            for mx in range(len(map)):
                for my in range(len(map[mx])):
                    if (a.x < b.x and (mx > a.x and mx < b.x)) or (a.x >= b.x and (mx <= a.x and mx >= b.x)):
                        if (a.y < b.y and (my > a.y and my < b.y)) or (a.y >= b.y and (my <= a.y and my >= b.y)):
                            if map[mx][my] == model.Tile.WALL:
                                if sign(a, b, my, mx) == sign(a, b, my + 1, mx) == sign(a, b, my + 1, mx + 1) == sign(a, b, my, mx + 1):
                                    continue
                                else:
                                    w.append(model.Vec2Float(mx, my))
            if w:
                mw = min(w, key=lambda m: distance_sqr(m, bullet.position))
                e = bullet.explosion_params.radius + 1
                if distance_sqr(unit.position, mw) < e:
                    targ = copy(target_pos)
                    p = copy(self.direction)
                    u = copy(self.direction)

                    o = si(unit.position.x - mw.x)
                    for i in range(int(e)):
                        if map[mw.x + (o*i)][mw.y] == model.Tile.WALL:
                            u = -1*self.direction
                    targ.x += u*e

                    return True
            return False
        danger_bullets = []
        targ = copy(target_pos)
        if game.bullets:
            enemy_bullets = filter(
                lambda u: u.player_id != unit.player_id, game.bullets)
            for bullet in enemy_bullets:
                # draw_bullet_path(bullet)
                if collision(unit.position, bullet):
                    danger_bullets.append(bullet)
                if bullet.weapon_type == model.WeaponType(2):
                    override = get_away(bullet, game.level.tiles)

            if danger_bullets:
                override = avoid_bullets(danger_bullets)
        if len(danger_bullets) == 0:
            override = False

        # End of Matrix Code
        if override:
            if nearest_health is not None:
                target_pos = copy(targ)
            elif distance_sqr(target_pos, unit.position) <= 4:
                target_pos = copy(targ)

        if self.override and override == False:
            target_pos = copy(self.target_pos)

        if target_pos.x != unit.position.x:
            jump = jump_calc(target_pos, unit, game)
            not_jump = not jump
        else:
            jump = False
            not_jump = not jump
            if target_pos.y != unit.position.y:
                not_jump = False
        debug.draw(model.CustomData.Log(
            "Jump: {}".format(jump)))
        vel = si(target_pos.x - unit.position.x) * \
            game.properties.unit_max_horizontal_speed

        self.prev = nearest_enemy.position
        self.tick += 1

        if target_pos.x > unit.position.x:
            self.direction = 1
        else:
            self.direction = -1
        self.dist = dist
        if override and self.target_pos is None:
            self.target_pos = target_pos
            self.override = override
        if self.target_pos:
            if (distance_sqr(self.target_pos, unit.position) < 0.5) or self.ttick >= 10:
                self.target_pos = target_pos
                self.override = override
                self.ttick = 0
        if self.override:
            self.ttick += 1
        debug.draw(model.CustomData.Log("Position: {}".format(unit.position)))
        debug.draw(model.CustomData.Log("A Fire: {}".format(self.allow_fire)))

        debug.draw(model.CustomData.Log("Target: {}".format(target_pos)))
        debug.draw(model.CustomData.Log("Distance x: {}".format(nearest_enemy.position.x - unit.position.x)))
        debug.draw(model.CustomData.Log("Distance y: {}".format(nearest_enemy.position.y - unit.position.y)))

        return model.UnitAction(
            velocity=vel,
            jump=jump,
            jump_down=not_jump,
            aim=aim,
            shoot=self.allow_fire,
            reload=False,
            swap_weapon=swap,
            plant_mine=False)
