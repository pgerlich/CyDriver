var async = require('async');
var gpio = require('pi-gpio');

/* GPIO Pins that we will use */
var pins = {
	motor1a: 16,
	motor1b: 18,
	motor1e: 22,
	motor2a: 21,
	motor2b: 19,
	motor2e: 23
};

var ready = false;

/* Returns an array of functions to be called with async.parallel */
function createTasks(obj, which) {
	return Object.keys(obj).map(function (key) {
		if (which === 'open') {
			return function (callback) {
				gpio.open(obj[key], "output", function (err) {
					if (err) console.log(err);
					callback();
				});
			};
		} else if (which === 'close') {
			return function (callback) {
				gpio.close(obj[key], function (err) {
					if (err) console.log(err);
					callback();
				});
			};
		} else if (which === 'write') {
			return function (callback) {
				gpio.write(parseInt(key), obj[key], function (err) {
					if (err) console.log(err);
					callback();
				});
			};
		}
	});
}

/* Set two pins to a specific value */
function setPins(pin1, pin2, value) {
	if (ready) {
		ready = false;
		var obj = {
			[pin1]: value,
			[pin2]: value
		};
		var tasks = createTasks(obj, 'write');
		async.parallel(tasks, function () {
			ready = true;
		});	
	}
}

module.exports = {
	init: function (cb) {
		var tasks = createTasks(pins, 'open');
		async.parallel(tasks, function () {
			ready = true;
			cb();
		});
	},
	destroy: function (cb) {
		var tasks = createTasks(pins, 'close');
		async.parallel(tasks, function () {
			ready = false;
			cb();
		});
	},
	forward: function (value) {
		setPins(pins.motor1a, pins.motor1e, value);
	},
	backward: function (value) {
		setPins(pins.motor1b, pins.motor1e, value);
	},
	left: function (value) {
		setPins(pins.motor2b, pins.motor2e, value);
	},
	right: function (value) {
		setPins(pins.motor2a, pins.motor2e, value);
	}
};