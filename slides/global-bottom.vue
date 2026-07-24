<script setup>
import { computed } from 'vue'
import { useNav } from '@slidev/client'

// Persistent section breadcrumb (option C). Each section's opening slide is
// tagged with a `section:` frontmatter; every following slide walks back to the
// nearest tag, so the current section stays visible — even when jumping via the
// skip-to-recap links.
const { slides, currentPage, currentLayout } = useNav()

const section = computed(() => {
  const all = slides?.value || []
  const idx = (currentPage?.value || 1) - 1
  for (let i = Math.min(idx, all.length - 1); i >= 0; i--) {
    const s = all[i]?.meta?.slide?.frontmatter?.section
    if (s) return s
  }
  return ''
})
</script>

<template>
  <footer v-if="currentLayout !== 'section' && currentLayout !== 'cover'" class="global-footer">
    <span v-if="section" class="section-crumb">{{ section }}</span>
    <span class="footer-right">
      <Link to="setup" class="setup-link">Setup</Link>
      <span class="slide-number">{{ currentPage }}</span>
    </span>
  </footer>
</template>

<style scoped>
.global-footer {
  position: absolute;
  bottom: 1rem;
  left: 1rem;
  right: 1rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.section-crumb {
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.4;
}

.footer-right {
  display: flex;
  align-items: center;
  gap: 0.9rem;
}

.setup-link {
  font-size: 0.7rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  opacity: 0.4;
  border-bottom: 1px dotted currentColor;
}

.slide-number {
  font-size: 0.7rem;
  opacity: 0.4;
}
</style>
