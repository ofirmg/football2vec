import os

DEBUG = True
PLOTLY_USERNAME = '?'
PLOTLY_API_KEY = '?'
ARTIFACTS = 'artifacts'
MODELS_ARTIFACTS = os.path.join(ARTIFACTS, 'models_artifacts')
DATA = 'data'


class PATHS:
    # INPUTS #
    # Data sources paths
    STATSBOMB_DATA = os.path.join(DATA, 'statsbomb', 'data')
    PLAYERS_IMAGES = os.path.join(DATA, 'pics', 'players')
    ATTRIBUTES_BADGES = os.path.join(DATA, 'pics', 'attributes_badges')

    # OUTPUTS #
    # Processed data paths
    ## Events data
    ENRICH_PLAYERS_METADATA_PATH = os.path.join(ARTIFACTS, 'enriched_player_metadata.pickle')
    ENRICH_EVENTS_DATA_PATH = os.path.join(ARTIFACTS, 'enriched_events_data.pickle')

    # Metadata and metrics paths
    PLAYERS_METADATA_PATH = os.path.join(ARTIFACTS, 'players_metadata.csv')
    TEAMS_METADATA_PATH = os.path.join(ARTIFACTS, 'teams_metadata.csv')
    MATCHES_METADATA_PATH = os.path.join(ARTIFACTS, 'matches_metadata.csv')
    PLAYERS_METRICS_PATH = os.path.join(ARTIFACTS, 'players_metrics_df.csv')
    BASELINE_PLAYERS_METRICS_PATH = os.path.join(ARTIFACTS, 'baselines_metrics.pickle')
    BASELINE_BY_SEASONS_METRICS_PATH = os.path.join(ARTIFACTS, 'baselines_metrics.pickle')
    PLAYERS_METRICS_BY_SEASON = os.path.join(ARTIFACTS, 'players_metrics_by_season.csv')

    # Explainers paths
    EXPLAINERS = os.path.join(ARTIFACTS, 'explainers')


class CONSTANTS:
    LINEUP_ACTION = 'Starting XI'
    CREDITS_TEXT = "Data by StatsBomb Open Dataset https://github.com/statsbomb/open-data"
    PITCH_DIMENSIONS = (106.0, 68)
    DEFAULT_MARKERS = ['Marker', 'Arrow']
    DEFAULT_SCATTER = 'o'

    # Plots constants
    FIGSIZE = (12, 8)
    LANDSCAPE_FIGSIZE = (8, 2)
    NARROW_FIGSIZE = (3, 6)
    TITLE_FONT_SIZE = 16
    AXIS_LABEL_FONT_SIZE = 14
    LEGEND_FONT_SIZE = 12
    TICKS_FONT_SIZE = 10
    ON_FIG_FONT_SIZE = 8
    ON_PITCH_FONT_SIZE = 6

    # Backgrounds
    PITCH_LAB_BACKGROUND = os.path.join(DATA, 'pics', 'pitch_lab.png')
    PITCH_BACKGROUND = os.path.join(DATA, 'pics', 'pitch.png')

    # Player placeholder (shown in the UI in case the player does not have an image in PATHS.PLAYERS_IMAGES
    PLAYER_IMAGE_PLACEHOLDER = "player_placeholder"

    # xG analysis
    HARD_XG = 0.05
    EASY_XG = 0.4

    # Language models parameters
    VOCABULARY = ['Ball Receipt*', 'Pressure', 'Dispossessed', 'Duel', 'Ball Recovery',
                  'Dribbled Past', 'Dribble', 'Interception', 'Block', 'Foul Committed',
                  'Foul Won', 'Goal Keeper', 'Miscontrol', 'Clearance',  # 'Shield',
                  'Offside', 'Pass', 'Shot', 'Carry', 'Goal Keeper']


class COLORS:
    LIGHT_GREEN = 'lightgreen'
    FIREBRICK = 'firebrick'
    WHITE = 'white'
    BLACK = 'black'
    DIMGREY = 'dimgrey'
    LIGHTGREY = 'lightgrey'
    DARK_GREEN = 'darkgreen'
    FOREST_GREEN = 'forestgreen'
    RED = 'red'
    BLUE = 'blue'
    W = 'w'
    K = 'k'


class PLAYERS:
    DIEGO_COSTA = 'Diego da Silva Costa'
    MANE = 'Sadio Mané'
    SALAH = 'Mohamed Salah'
    LUIS_SUAREZ = 'Luis Alberto Suárez Díaz'
    MESSI = 'Lionel Andrés Messi Cuccittini'
    SERGIO_RAMOS = 'Sergio Ramos García'
    PIQUE = 'Gerard Piqué Bernabéu'
    DE_JONG = 'Frenkie de Jong'
    BENZEMA = 'Karim Benzema'
    GRIEZMANN = 'Antoine Griezmann'
    DEMBELE = 'Ousmane Dembélé'
    BUSQUETS = 'Sergio Busquets i Burgos'
    LENGLET = 'Clément Lenglet'
    JORDI_ALBA = 'Jordi Alba Ramos'
    MARCELO = 'Marcelo Vieira da Silva Júnior'
    HENRY = 'Thierry Henry'
    NEYMAR = 'Neymar da Silva Santos Junior'
    XAVI = 'Xavier Hernández Creus'
    FATI = 'Anssumane Fati'
    INIESTA = 'Andrés Iniesta Luján'
    GIROUD = 'Olivier Giroud'
    WERNER = 'Timo Werner'
    INAKI_WILIAMS = 'Iñaki Williams Arthuer'
    ETOO = 'Samuel Et"o Fils'
    COUTINHO = 'Philippe Coutinho Correia'
    DE_BRUYNE = 'Kevin De Bruyne'
    KROOS = 'Toni Kroos'
    TER_STEGEN = 'Marc-André ter Stegen'
    GABRIEL_JESUS = 'Gabriel Fernando de Jesus'
    STERLING = 'Raheem Shaquille Sterling'
    SERGIO_ROBERTO = 'Sergi Roberto Carnicer'
    THIAGO_ALCANTARA = 'Thiago Alcântara do Nascimento'
    RAFINHA_ALCANTARA = 'Rafael Alcântara do Nascimento'
    ZLATAN = 'Zlatan Ibrahimović'
    RAKITIC = 'Ivan Rakitić'


SHORTNAMES = {
    'Gabriel Fernando de Jesus': 'G.Jesus',
    'Samuel Et"o Fils': "Samuel Eto'o",
    'Sadio Mané': 'Mané',
    'Olivier Giroud': 'Giroud',
    'Timo Werner': 'Werner',
    'Iñaki Williams Arthuer': 'Iñaki.W',
    'Mohamed Salah': 'Salah',
    'Luis Alberto Suárez Díaz': 'Suárez',
    'Lionel Andrés Messi Cuccittini': 'Messi',
    'Sergio Ramos García': 'S. Ramos',
    'Gerard Piqué Bernabéu': 'Piqué',
    'Frenkie de Jong': 'De Jong',
    'Karim Benzema': 'Benzema',
    'Antoine Griezmann': 'Griezmann',
    'Ousmane Dembélé': 'Dembélé',
    'Sergio Busquets i Burgos': 'Busquets',
    'Clément Lenglet': 'Lenglet',
    'Jordi Alba Ramos': 'J. Alba',
    'Marcelo Vieira da Silva Júnior': 'Marcelo',
    'Thierry Henry': 'Henry',
    'Neymar da Silva Santos Junior': 'Neymar',
    'Xavier Hernández Creus': 'Xavi',
    'Anssumane Fati': 'Fati',
    'Andrés Iniesta Luján': 'Iniesta',
    'Philippe Coutinho Correia': 'Countinho',
    'Kevin De Bruyne': 'De Bruyne',
    'Toni Kroos': 'Kroos',
    'Marc-André ter Stegen': 'Ter-Stegen',
    'Raheem Shaquille Sterling': 'Sterling',
    'Sergi Roberto Carnicer': 'S.Roberto',
    'Thiago Alcântara do Nascimento': 'Thiago',
    'Rafael Alcântara do Nascimento': 'Rafinha',
    'Zlatan Ibrahimović': 'Ibrahimović',
    'Ivan Rakitić': 'Rakitić'}


class COLUMNS:
    IS_SHOT = 'is_shot'
    SHOOTING = 'shooting'
    TEAM_MANAGERS = 'team_managers'
    TEAM_GENDER = 'team_gender'
    ONE_ON_ONE = 'one_on_one'
    COUNTRY_NAME = 'country_name'
    POSSESSION = 'possession'
    HEADER = 'header_shot'
    PENALTY = 'penalty_shot'
    FREE_KICK = 'free_kick_shot'
    OUTBOX_SHOT = 'out_of_box_shot'
    DRIBBLE_WON = 'dribble_won'
    GOAL = 'goal'
    ASSISTS = 'pass_goal_assist'
    XA = 'xA'
    LOCATION = 'location'
    START_X = 'POS_START_X'
    START_Y = 'POS_START_Y'
    END_X = 'POS_END_X'
    END_Y = 'POS_END_Y'
    ACTION_TYPE = 'type_name'
    TEAM_NAME = 'team_name'
    TEAM_SIDE = 'Team'
    PLAYER_NAME = 'player_name'
    XG = 'shot_statsbomb_xg'
    MATCH_ID = 'match_id'
    SEASON_NAME = 'season_name'
    COMPETITION_NAME = 'competition_name'
    HALF = 'period'
    TIME = 'timestamp'
    POSITION = 'position_name'


SKILLS = [COLUMNS.SHOOTING, COLUMNS.HEADER, COLUMNS.ONE_ON_ONE, COLUMNS.PENALTY,
          COLUMNS.OUTBOX_SHOT, COLUMNS.FREE_KICK, COLUMNS.XA, COLUMNS.DRIBBLE_WON]


class BADGES:
    # Badges
    PATHS = {COLUMNS.SHOOTING: os.path.join(PATHS.ATTRIBUTES_BADGES, 'finishing.png'),
             COLUMNS.HEADER: os.path.join(PATHS.ATTRIBUTES_BADGES, 'heading.png'),
             COLUMNS.XA: os.path.join(PATHS.ATTRIBUTES_BADGES, 'assists.png'),
             COLUMNS.FREE_KICK: os.path.join(PATHS.ATTRIBUTES_BADGES, 'free_kick.png'),
             COLUMNS.OUTBOX_SHOT: os.path.join(PATHS.ATTRIBUTES_BADGES, 'long-range.png'),
             COLUMNS.DRIBBLE_WON: os.path.join(PATHS.ATTRIBUTES_BADGES, 'dribbling.png'),
             COLUMNS.ONE_ON_ONE: os.path.join(PATHS.ATTRIBUTES_BADGES, 'one_on_one.png')}

    CAPTIONS = {
        COLUMNS.SHOOTING: "Deadly shooting",
        COLUMNS.ONE_ON_ONE: "1-on-1 finisher",
        COLUMNS.HEADER: "Superb headers",
        COLUMNS.FREE_KICK: "Free kicks specialist",
        COLUMNS.OUTBOX_SHOT: "Long range sniper",
        COLUMNS.DRIBBLE_WON: "Super dribbler",
        COLUMNS.XA: "Top chances creator"
    }


class ANALYSIS_PARAMS:
    DEFAULT_XG_METRICS = [f'{COLUMNS.SHOOTING}:LIFT:percentile',
                          f'{COLUMNS.HEADER}:LIFT:percentile',
                          f'{COLUMNS.ONE_ON_ONE}:LIFT:percentile',
                          f'{COLUMNS.OUTBOX_SHOT}:LIFT:percentile',
                          f'{COLUMNS.FREE_KICK}:LIFT:percentile',
                          f'{COLUMNS.XA}:mean:percentile',
                          f'{COLUMNS.DRIBBLE_WON}:mean:percentile']

    DEFAULT_XG_METRICS_LABELS = [f'{COLUMNS.SHOOTING} Lift percentile',
                                 f'{COLUMNS.HEADER} Lift percentile',
                                 f'{COLUMNS.ONE_ON_ONE} Lift percentile',
                                 f'{COLUMNS.OUTBOX_SHOT} Lift percentile',
                                 f'{COLUMNS.FREE_KICK} Lift percentile',
                                 f'{COLUMNS.XA} percentile',
                                 f'{COLUMNS.DRIBBLE_WON} percentile']

    BASELINES_NAMES = {COLUMNS.COMPETITION_NAME: 'Competition baseline',
                       COLUMNS.POSITION: 'Position baseline'}

    BASELINES_TO_USE = {COLUMNS.POSITION: ['Center Forward', 'Left Center Forward', 'Left Wing', 'Right Wing'],
                        COLUMNS.COMPETITION_NAME: ['La Liga', 'FIFA World Cup', 'Premiere League']}

