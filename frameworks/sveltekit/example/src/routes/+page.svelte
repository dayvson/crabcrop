<script lang="ts">
  // Renders one image through the crabcrop loader from
  // $lib/images. The page names no crabcrop operations: the
  // "card" preset on the server carries the policy, and the
  // loader appends only the widths below.
  import { images } from "$lib/images";

  const photo = {
    url: "https://images.unsplash.com/photo-1502680390469-be75c86b636f",
    alt: "A surfer riding a wave",
  };
</script>

<h1>crabcrop for SvelteKit</h1>
<img
  src={images.url(photo.url, { width: 800 })}
  srcset={images.srcset(photo.url, [320, 640, 800])}
  sizes="(min-width: 800px) 800px, 100vw"
  alt={photo.alt}
/>
