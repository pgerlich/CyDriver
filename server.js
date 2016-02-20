var express = require('express');
var port = 8080;
var http = require("http");
var vidStreamer = require("vid-streamer");
var path = require('path');

var app = express();
app.set('view engine', 'jade');

app.get('/', function (req, res) {
    res.sendFile(path.join(__dirname, '/src', 'test.html'));
});

var newSettings = {
    rootPath: "videos/",
    forceDownload: false
}

app.get("/videos", vidStreamer.settings(newSettings));

app.listen(port, function () {
    console.log('Example app listening on port '+port);
});
