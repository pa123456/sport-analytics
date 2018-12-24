 <!DOCTYPE html>
<html>
<head>
  <title>Upload your files</title>
</head>
<body>
  <form enctype="multipart/form-data" action="index.php" method="POST">
    <p>Upload your file</p>
    <p>Insert side-view video here</p>
    <input type="file" name="uploaded_side"></input><br />
    <p>Insert front-view video here</p>
    <input type="file" name="uploaded_front"></input><br />
    <p>Upload videos to server</p>
    <input type="submit" value="Upload"></input>
    <p>Process videos</p>
    <button type="submit" name="run">Process videos</button>
    <a href="./display_processed.html">Display Processed Videos</a>
  </form>
</body>
</html>

