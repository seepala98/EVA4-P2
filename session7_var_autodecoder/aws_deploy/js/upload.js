function uploadAndClassifyImage(url){
	var fileInput = document.getElementById('resnet34FileUpload').files;
	if(!fileInput.length){
		return alert('Please choose a file to upload first');
	}
	
	var file = fileInput[0];
	var filename = file.name;
	
	var formData = new FormData();
	formData.append(filename, file);
	
	console.log(filename);
	console.log(url);
	
	$.ajax({
		async: true,
		crossDomain: true,
		method: 'POST',
		url: url,
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})
	.done(function (response) {
		responseJson = JSON.parse(response);
		console.log(responseJson.predicted);
		document.getElementById('result').textContent = responseJson.predicted;
		document.getElementById("ItemPreview").src = "data:image/png;base64," + response;
	})
	.fail(function (error) {
		alert("There was an error while sending prediction request to resnet34 model."); 
		console.log(error);
	});
};

function generateCarImage(url){

$.ajax({
      async: false,
      crossDomain: true,
      method: 'POST',
      url: url,
      processData: false,
      contentType: false,
})
.done(function(response){
  //response = JSON.parse(response);
  $('#ganResult').attr('src', 'data:image/jpeg;base64,'+response.img);
//$('#faceResult').attr('src', response);
})
//.fail(function() {alert ("There was an error while sending request to GAN service."); });
};


$('#btnGenerateCarImage').click(generateCarImage);

function uploadcarvae(){
	var fileInput = document.getElementById('ReconstructCarUpload').files;
	if(!fileInput.length){
		return alert('please choose file to upload first');
	}
	
	var file = fileInput[0];
	var filename = file.name;

	var formData = new FormData();
	formData.append(filename, file);

	console.log(filename);

	$.ajax({
		async: true,
		crossDomain: true,
		method: 'POST',
		url: 'https://cqzkp9tfz7.execute-api.ap-south-1.amazonaws.com/dev/getcars',
		data: formData,
		processData: false,
		contentType: false,
		mimeType: "multipart/form-data",
	})	
	.done(function(response){
		console.log(response);
		var image = document.getElementById('output_7_');
		image.src = "https:///eva4p2-session1.s3.amazonaws.com/cars.jpg?t=" + new Date().getTime();
	})
	.fail(function(){alert("There was an error while sending prediction request to align model");});
	};
	
    $('#btnReconstructCarUpload').click(uploadcarvae);

//$('#btnResNetUpload').click(uploadAndClassifyImage);