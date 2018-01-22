const express = require('express');
const app = require('express')();
const http = require('http').Server(app);
const io = require('socket.io')(http);
const spawn = require('child_process').spawn;
const sh = spawn('./dm.o', ['-d=0', '-c=cal.yml']);


app.use(express.static('.'))
// app.get('/', function(req, res){
//   res.sendFile(__dirname + '/index.html');
// });

io.on('connection', function(socket){
  console.log('a user connected');
});

sh.stdout.on('data', function(data) {
  console.log( data.toString());
  io.emit('message', data.toString());
});

sh.stderr.on('data', function(data) {
  console.log( data.toString());
  io.emit('message', data.toString());
});

sh.on('exit', function (code) {
  console.log('exit', '** Shell exited: '+code+' **');
  io.emit('exit', '** Shell exited: '+code+' **');
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});
    
