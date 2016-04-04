'''
Borrowed from: https://github.com/sousys/ogs/blob/master/ogs.py
AND from: https://gist.github.com/apiarian/3bf841bb3681351dcb5198bc40249ba8
'''

import requests   # Install this module with "pip install requests" 
from ogs_credentials import *  # imports client_id, client_secret, username, password
from urllib2 import Request, urlopen
  
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


def rank_to_display(rank_value):
    if rank_value < 30:
        return '%dk' % (30-rank_value,)
    else:
        return '%dd' % ((rank_value-30)+1)

# id is id of the challenge 
def get_challenge_details(id):

    request = Request('http://online-go.com/v1/challenges/id')
    
    response_body = urlopen(request).read()
    print response_body
    
def delete_challenge():
    request = Request('http://online-go.com/v1/challenges/id')
    request.get_method = lambda: 'DELETE'

    response_body = urlopen(request).read()
    print response_body

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

    print(r.request.method, r.request.url, r.request.body);

    oauth2_json = r.json()
    access_token = oauth2_json['access_token']
    refresh_token = oauth2_json['refresh_token']

    print(oauth2_json)
    print()
    return access_token

# get my info
def get_my_info(s, access_token):

    r = s.get('https://online-go.com/api/v1/me/',
              headers = {
                         'Authorization': 'Bearer {}'.format(access_token),
                         },
              )

    my_info_json = r.json()
    print(my_info_json)
    print()
    print('overall: {}, blitz: {}, live: {}, corr: {}'.format(
                                                              rank_to_display(my_info_json['ranking']),
                                                              rank_to_display(my_info_json['ranking_blitz']),
                                                              rank_to_display(my_info_json['ranking_live']),
                                                              rank_to_display(my_info_json['ranking_correspondence']),
                                                              ))
    print()
    return my_info_json

# s should be a requests.Session() object
def create_match2(s, access_token, my_info_json):

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
                                                            "main_time": 600,
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
    if 'status' in r.content.keys():
        if r.content['status'] == 'ok':
            return r.content
    return False

if __name__ == "__main__":
    # start a session
    s = requests.Session()
    # get access token
    access_token = connect(s)
    # get my info
    my_info_json = get_my_info(s, access_token)
    # post a challenge
    game_data = create_match2(s, access_token, my_info_json)
    # delete that challenge
    
    
    
    
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
