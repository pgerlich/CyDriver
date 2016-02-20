var car = require('./car');

function moveCar(keyCode, value) {
	switch (keyCode) {
		case 87:
			car.forward(value);
			break;
		case 65:
			car.left(value);
			break;
		case 83:
			car.backward(value);
			break;
		case 68:
			car.right(value);
			break;
	}
}

var clients = [];

module.exports = function (server) {
	
	var io = require('socket.io')(server);

	io.on('connection', function (socket) {
		if (clients.length === 0) {
			console.log('New connection');
			clients.push(socket);
			car.init(function () {
				socket.emit('ready', {});
			});
			socket.on('keydown', function (data) {
				moveCar(data.key, 1);
			});
			socket.on('keyup', function (data) {
				moveCar(data.key, 0);
			});
			socket.on('disconnect', function() {
				car.destroy(function () {
					for (var i = 0; i < clients.length; i++) {
						delete clients[i];
					}
					console.log('Disconnection');
				});
			});
		} else {
			console.log('A client attempted to connect, but was refused, as we already have a connected client');
		}
	});
};