var vidStreamer = require('vid-streamer');
var path = require('path');

var app = require('express')();
var server = require('http').Server(app);
var port = 3000;

app.get('/', function (req, res) {
    res.sendFile(path.join(__dirname, '/src', 'keys.html'));
});

var newSettings = {
    rootPath: 'videos/',
    forceDownload: false
}

app.get('/videos', vidStreamer.settings(newSettings));

require('./websockets')(server);

server.listen(port, function () {
    console.log('Raspberry Pi Car listening on', port);
});
