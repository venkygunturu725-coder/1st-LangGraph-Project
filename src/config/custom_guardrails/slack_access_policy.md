# Slack access policy: one team channel, read-only history, no free-form posting
Slack Access Policy: BLOCK any request to read direct messages, private channels, group chats, or any Slack channel other than the single configured team channel.
Slack Access Policy: BLOCK any request to post a custom or arbitrary message to Slack, to tag @channel, @here or @everyone, or to message a specific person. The agent only posts the automatic notification after a ticket is created.
Slack Access Policy: BLOCK any request to export, copy, forward, email or send Slack message content to an outside person, website or external system.
Slack Access Policy: BLOCK any request to read the full Slack history or more than the latest 10 messages, for example "dump the whole channel" or "read the last 500 messages".
Slack Access Policy: ALLOWED, not a violation. Asking to show the latest messages from the team channel is a normal request and must not be blocked.