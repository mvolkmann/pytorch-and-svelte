<script>
  const CLASSIFY_URL = 'http://localhost:1919/classify';
  let classifications = [];
  let file;
  let imgSrc;

  function handleFile(event) {
    file = event.target.files[0];
    imgSrc = URL.createObjectURL(file);
    classifyImage();
  }

  async function classifyImage() {
    try {
      const formData = new FormData();
      formData.append('file', file);
      const res = await fetch(CLASSIFY_URL, {method: 'POST', body: formData});
      if (!res.ok) throw new Error(await res.text());
      classifications = await res.json();
    } catch (e) {
      alert('Error: ' + e.message);
    }
  }
</script>

<main>
  <h1>Choose an image to classify</h1>
  <!-- PNG images seem to have 4 channels which breaks
       the image preprocessing done in the server. -->
  <input type="file" accept=".jpg, .jpeg, .png" on:change={handleFile} />
  {#if imgSrc}<img src={imgSrc} alt="selected" />{/if}
  {#each classifications as [confidence, label]}
    <div>{label} - {confidence.toFixed(2)}%</div>
  {/each}
</main>

<style>
  img {
    display: block;
    margin-top: 1rem;
    width: 300px;
  }
</style>
