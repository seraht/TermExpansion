"""
Authors: ST & MM
UMLS Authentication
"""
# See https://documentation.uts.nlm.nih.gov/rest/authentication.html for full explanation

import json
import requests
from lxml.html import fromstring

class Authentication:
    def __init__(self, apikey):
        """
        Getting the apikey for the specific api
        :param apikey:
        """
        self.apikey = apikey
        self.URI = "https://utslogin.nlm.nih.gov"
        self.AUTH_ENDPOINT = "/cas/v1/api-key"
        self.apikey = 'c8352956-d42f-4369-bd81-5936c34a3994'
        self.service = "http://umlsks.nlm.nih.gov"

    def gettgt(self):
        """
        Authorization
        Http headers and request for UML api
        :return:
        """
        params = {'apikey': self.apikey}
        h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
        r = requests.post(self.URI + self.AUTH_ENDPOINT, data=params, headers=h)
        response = fromstring(r.text)
        tgt = response.xpath('//form/@action')[0]
        return tgt

    def getst(self, tgt):
        """
        Call api thanks to tgt
        Http headers and request for UML api
        :param tgt:
        :return:
        """
        params = {'service': self.service}
        h = {"Content-type": "application/x-www-form-urlencoded", "Accept": "text/plain", "User-Agent": "python"}
        r = requests.post(tgt, data=params, headers=h)
        st = r.text
        return st


def get_synonym(apikey, finding):
    """
    Retrieve all the words expansion of a word using the UMLS API
    :param apikey: api key
    :param finding: Finding name, term to look for
    :return: Synonyms for a given term
    """
    syn = []
    auth_client = Authentication(apikey)
    tgt = auth_client.gettgt()
    ticket = auth_client.getst(tgt)

    uri = "https://uts-ws.nlm.nih.gov/rest"
    r = requests.get(f'{uri}/search/current?string={finding}&ticket={ticket}&searchType=approximate')
    r.encoding = 'utf-8'
    items = json.loads(r.text)
    for synonym in items['result']['results']:
        if synonym['name'].lower() not in finding.lower():
            syn.append(synonym['name'])
    return syn
