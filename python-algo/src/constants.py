ARENA_SIZE = 28
LOCATION_HEAD_SIZE = 210
MY_LOCATIONS = [[x, y] for y in range(14) for x in range(13-y, 15+y)]

# actions
ACTION_SHORTHAND = ['NOOP', 'FF', 'EF', 'DF', 'PI', 'EI', 'SI', 'RM', 'UP']
NOOP        = 0
WALL        = 1
SUPPORT     = 2
TURRET      = 3
SCOUT       = 4
DEMOLISHER  = 5
INTERCEPTOR = 6
REMOVE      = 7
UPGRADE     = 8
STRUCTURES  = [WALL, SUPPORT, TURRET]
MOBILES     = [SCOUT, DEMOLISHER, INTERCEPTOR]