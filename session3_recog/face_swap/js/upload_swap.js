function uploadImagefaceswap(url){
	var fileInput = document.getElementById('resnet34FileUpload').files;
	var fileInput2 = document.getElementById('resnet34FileUpload2').files;

	if(!fileInput.length){
		return alert('Please choose a file to upload first');
	}

	if(!fileInput2.length){
		return alert('Please choose a file to upload first');
	}

	var file = fileInput[0];
	var filename = file.name;
	
	var file2 = fileInput2[0];
	var filename2 = file2.name;

	var formData = new FormData();
	formData.append(filename, file);
	formData.append(filename2, file2);
	
	console.log(filename);
	console.log(filename2);
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
		document.getElementById("ItemPreview").src = "data:image/png;base64," + response;
	})
	.fail(function (error) {
		alert("There was an error while sending prediction request to resnet34 model."); 
		console.log(error);
	});
};

//$('#btnResNetUpload').click(uploadAndClassifyImage);