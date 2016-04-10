'''
Borrowed from: https://github.com/sousys/ogs/blob/master/ogs.py
AND from: https://gist.github.com/apiarian/3bf841bb3681351dcb5198bc40249ba8
'''

import requests   # Install this module with "pip install requests" 
from ogs_credentials import *  # imports client_id, client_secret, username, password
from urllib2 import Request, urlopen
import time
import json
from socketIO_client import SocketIO
from base64 import b64encode
import string

def get_token(client_id, client_secret, username, password):
    '''
    The API documentation was a bit confusing since it refered to urls that were http instead of https
    and returned an error so keep that in mind when reading the api documentation.
    Also the 'data='-part of the post-request is not really needed but i included it in the request for clarity since this is how the examples in 
    the requests documentation show it. 
    '''
    url = "https://online-go.com/oauth2/access_token"
    ogs_response = requests.post(url, data={"grant_type" : "password", "client_id" : client_id, "client_secret" : client_secret, "username" : username, "password" : password})
    return ogs_response.json()["access_token"]

  
'''
Below are 3 simple examples of how the token then is used to get data from the API. All very simple.
  
'''
def get_user_vitals(token):
    url = "https://online-go.com/api/v1/me/"
    vitals = requests.get(url, headers={"Authorization" : "Bearer " + token})
    return vitals.json()

def get_user_settings(token):
    url = "https://online-go.com/api/v1/me/settings"
    settings = requests.get(url, headers={"Authorization" : "Bearer " + token})
    return settings.json()

def get_user_games(token):
    url = "https://online-go.com/api/v1/me/games"
    games = requests.get(url, headers={"Authorization" : "Bearer " + token})
    return games.json()  


def create_a_match(token):
    from urllib2 import Request, urlopen
    url = "http://online-go.com/v1/challenges/"
    headers = {"Authorization" : "Bearer " + token, 'Content-Type': 'application/json'}
    body = """ {"game": {"name": "friendly match 123",
                     "rules": "japanese",
                     "ranked": false,
                     "handicap": 0,
                     "time_control_parameters": {
                                                  "time_control": "fischer",
                                                  "initial_time": 259200,
                                                  "max_time": 604800,
                                                  "time_increment": 86400
                                                  },
                     "pause_on_weekends": false,
                     "width": 9,
                     "height": 9,
                     "disable_analysis": true
                     },
            "challenger_color": "automatic",
            "min_ranking": 0,
            "max_ranking": 0
            }"""
    #match = requests.post(url, data=body, headers=headers)
    request = Request('http://online-go.com/v1/challenges/', data=body, headers=headers)
    response_body = urlopen(request).read()
    print("response_body = "+str(response_body))
#     try:
#         pass
#         
#         print("dir(match) = "+str(dir(match)))
#         print("dir(match.json) = "+str(dir(match.json)))
#         for item in dir(match):
#             if item == 'content':
#                 with open('response.html', 'w') as f:
#                     f.write(str(match.content))
#             if '_' not in item and item != 'content':
#                 print("  match."+str(item)+" " +str(getattr(match,item)))
#                 if item == 'json':
#                     for inner_item in dir(match.json):
#                         if '__' not in inner_item:
#                             print("    match.json."+str(inner_item)+" " +str(getattr(match.json,inner_item)))
#                         if inner_item == 'im_self':
#                             for inner2_item in dir(match.json.im_self):
#                                 if '__' not in inner2_item and inner2_item != 'content':
#                                     #print("      match.json.im_self."+str(inner2_item)+" " +str(getattr(match.json.im_self,inner2_item)))
#                                     pass
#         #print("match.content = "+str(match.content))
#         #print("match.json = "+str(match.request))
#         
#         #print response_body
#     except:
#         pass
#     return match
    return response_body


def submit_move_to_game(s,access_token,player_id,game_id,move):
    #url = 'http://online-go.com/v1/games/'+str(game_id)+'/'+str(move)+'/'
    url = 'https://online-go.com/api/v1/me/games/'+str(game_id)+'/'+str(move)+'/'
    r = s.post(url,
               headers = {
                          'Authorization': 'Bearer {}'.format(access_token),
                          },
               json = {
                       'game_id': game_id,
                       'player_id': player_id,
                       'move': move,
                        },
               allow_redirects = False,
               );
                
#     r = s.get('http://online-go.com/v1/games/'+str(game_id)+'/'+str(move)+'/',
#               headers = {
#                          'Authorization': 'Bearer {}'.format(access_token),
#                          },
#               )
#                
    print(r.content)
    if r.content:
        results = json.loads(r.content)    
        return results
    else:
        return False

def rank_to_display(rank_value):
    if rank_value < 30:
        return '%dk' % (30-rank_value,)
    else:
        return '%dd' % ((rank_value-30)+1)

# id is id of the challenge 
def get_challenge_details(s,access_token, id):
    r = s.get('https://online-go.com/api/v1/me/challenges/'+str(id),
              headers = {
                         'Authorization': 'Bearer {}'.format(access_token),
                         },
              )

#     print(r.request.method, r.request.url, r.request.body)
#     print(r.content)

def get_game_auths(s,access_token,game_id):
    '''
    Returns both the game auth and the game chat auth for the given game_id
    '''
    r = s.get('https://online-go.com/api/v1/games/'+str(game_id)+'/',
               headers = {
                          'Authorization': 'Bearer {}'.format(access_token),
                          },
               allow_redirects = False,
               );

    #print("getting back: ")
    #print(r.request.method, r.request.url, r.request.headers, r.request.body)
    #print()
    #print("headers are:")
    #print(r.headers)
    #print("content is: ")
    #print(r.content)
    results = json.loads(r.content) # turn raw json str into python dict
    #print("results.keys() are "+str(results.keys()))
    #print("auth key is "+str(results['auth']))
    #print("chat auth key is "+str(results['game_chat_auth']))
    if 'auth' in results.keys() and 'game_chat_auth' in results.keys():
        return results['auth'], results['game_chat_auth']
    

def delete_challenge(s, access_token, id):
    url1 = 'https://online-go.com/api/v1/me/challenges/'+str(id)+'/'
    url2 = 'http://online-go.com/v1/challenges/'+str(id)+'/'
    r = s.delete(url2,
              headers = {
                         'Authorization': 'Bearer {}'.format(access_token),
                         },
              )
    
    print(r.request.method, r.request.url, r.request.body)
    print(r.content)
    #request = Request('http://online-go.com/v1/challenges/'+str(id))
    #request.get_method = lambda: 'DELETE'

    #response_body = urlopen(request).read()
    #print response_body
    
# returns the access token
def connect(s):
    d = {
         'client_id': client_id,
         'client_secret': client_secret,
         'grant_type': 'password',
         'username': username,
         'password': password,
         }

    r = s.post('https://online-go.com/oauth2/access_token',
               headers = {
                          'Content-Type': 'application/x-www-form-urlencoded',
                          },
               data = d,    
               allow_redirects = False,
               );

    #print(r.request.method, r.request.url, r.request.body)

    oauth2_json = r.json()
    access_token = oauth2_json['access_token']
    refresh_token = oauth2_json['refresh_token']

    #print(oauth2_json)
    #print()
    return access_token

# get my info
def get_my_info(s, access_token):

    r = s.get('https://online-go.com/api/v1/me/',
              headers = {
                         'Authorization': 'Bearer {}'.format(access_token),
                         },
              )

    my_info_json = r.json()
#     print(my_info_json)
#     print()
#     print('overall: {}, blitz: {}, live: {}, corr: {}'.format(
#                                                               rank_to_display(my_info_json['ranking']),
#                                                               rank_to_display(my_info_json['ranking_blitz']),
#                                                               rank_to_display(my_info_json['ranking_live']),
#                                                               rank_to_display(my_info_json['ranking_correspondence']),
#                                                               ))
#     print()
    return my_info_json

# s should be a requests.Session() object
def create_challenge(s, access_token, my_info_json):

    r = s.post('https://online-go.com/api/v1/challenges',
               headers = {
                          'Authorization': 'Bearer {}'.format(access_token),
                          },
               json = {
                       'game': {
                                'name': 'Friendly Go 123',
                                'rules': 'japanese',
                                'ranked': True,
                                'handicap': -1,
                                'time_control': 'byoyomi',
                                'time_control_parameters': {
                                                            "time_control": "byoyomi",
                                                            "main_time": 6000,
                                                            "period_time": 60,
                                                            "periods":5,
                                                            },
                                'pause_on_weekends': False,
                                'width': 19,
                                'height': 19,
                                'disable_analysis': True,
                                },
                       'challenger_color': 'automatic',
                       'min_ranking': my_info_json['ranking']-3,
                        'max_ranking': my_info_json['ranking']+3,
                        },
               allow_redirects = False,
               );

# print(r.request.method, r.request.url, r.request.headers, r.request.body)
# print()
#
# print(r.headers)
    print(r.content)
    results = json.loads(r.content) # turn raw json str into python dict
    if 'status' in results.keys():
        if results['status'] == 'ok':
            return results
    return False

def post_challenge():
    # start a session
    s = requests.Session()
    # get access token
    access_token = connect(s)
    # get my info
    my_info_json = get_my_info(s, access_token)
    # post a challenge
    game_data = create_challenge(s, access_token, my_info_json)
    challenge_id = game_data['challenge']
    game_id = game_data['game']
    print("challenge id is "+str(challenge_id))
    # delete that challenge
    time.sleep(0.5)
    challenge_deets = get_challenge_details(s,access_token,challenge_id)
    time.sleep(5)


def send_chat():
    #42["game/chat",{"auth":"f664b33c576ab01de7b65ce900ca7552","player_id":86412,"username":"dustyd","body":"hey are you getting this","ranking":19,"ui_class":"timeout","type":"discussion","game_id":4800859,"is_player":1,"move_number":2}]
    pass

def submit_move(s, access_token, g_id, player_id, move):
    '''
    Submitting moves happens using the real-time api, for which we will use socketIO-client
    '''
    
    ### FIRST GET ACCESS TOKEN
    # start a session
#     s = requests.Session()
#     # get access token
#     access_token = connect(s)
#     g_id = '4830343'
    game_auth, game_chat_auth = get_game_auths(s, access_token, g_id)
    
    ### NOW CONNECT VIA REAL TIME API
    # try connecting
    
    def on_response(*args):
        print('  Response after submitting move: ', args)

    with SocketIO('https://ggs.online-go.com/socket.io') as socketIO:
        #print("socket is connected: "+str(socket.connected))
        socketIO.emit("game/connect", 
                    {'game_id': g_id, 'player_id': player_id, 'chat': 1, 'game_type': "game", 'auth':game_auth},)
        socketIO.wait(1)
         
        # try submitting a move
        socketIO.emit("game/move",{"auth":game_auth,"game_id":g_id,"player_id":player_id,"move":move},
                      on_response)
        socketIO.wait_for_callbacks(seconds=3)
         
        
    
    #socket.emit("game/connect", {'game_id': 4800859, 'player_id': 86412, 'chat': 1, 'game_type': "game",
                                 #'auth':access_token})
    #socket.emit("game/move", {})
    #print("socket is connected: "+str(socket.connected))
    #socket.emit("game/connect", {game_id: 123, player_id: 1, chat: true});
    #socket.emit("game/connect",)
    
#     my_player_id = "86412"
#     game_id = "4792290"
#     move = 'bc'
#     move_results = submit_move_to_game(s,access_token,game_id,my_player_id,move)
#     print("just tried to submit move")
#     print("results are: "+str(move_results))
#     

def check_valid_move(mv):
    if len(mv) != 2:
        return False
    if mv[0] not in string.ascii_lowercase or mv[1] not in string.ascii_lowercase:
        return False
    return True

if __name__ == "__main__":
    # start a session
    print("starting session...")
    s = requests.Session()
    # get access token
    print("getting REST API access token...")
    access_token = connect(s)
    # get my info
    print("getting my player info via REST API...")
    my_info_json = get_my_info(s, access_token)
    my_player_id = my_info_json['id']
    # post a challenge
    print("posting a challenge with my access_token...")
    game_data = create_challenge(s, access_token, my_info_json)
    print("waiting for someone to accept challenge")
    time.sleep(5)
    challenge_id = game_data['challenge']
    game_id = game_data['game']
    #print("challenge id is "+str(challenge_id))
    # delete that challenge
    #time.sleep(0.5)
    #print("getting the challenge details")
    #challenge_deets = get_challenge_details(s,access_token,challenge_id)
    #time.sleep(5)
    print("Everything seems good, lets play!!")
    print("Please enter your desired move:")
    usr_input = raw_input()
    while usr_input != 'resign':
        if check_valid_move(usr_input):
            submit_move(s, access_token,game_id,my_player_id,usr_input)
        usr_input = raw_input()
    
    
    
    
    #post_challenge()


#     token = get_token(client_id, client_secret, username, password)
# 
#     print "\n--- vitals ---"
#     vitals = get_user_vitals(token)
#     for k in vitals.keys():
#         print k, ":", vitals[k]            #prints the keys and their value in the dict returned by get_user_vitals()
#         
#     print "\n--- settings ---"
#     settings = get_user_settings(token)
#     for k in settings.keys():
#         print k
#     for e in settings[k]:
#         print "  ", e, ":", settings[k][e]  #prints the keys and their value in the dict returned by get_user_settings().         
#   
#     print "\n--- games ---"
#     games = get_user_games(token)
#     for k in games.keys():
#         print k                       
#         
