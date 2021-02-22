from slack_webhook import Slack

def send_message(slack_hook, text):
    url = "https://hooks.slack.com/services/" + slack_hook
    slack = Slack(url=url)
    slack.post(text=text)