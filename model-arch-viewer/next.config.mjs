/** @type {import('next').NextConfig} */
const nextConfig = {
  // Allow serving subplot PNGs from the benchmarks directory
  images: {
    unoptimized: true,
  },
  // Hide the Next.js dev indicator (floating circle) from screenshots
  devIndicators: false,
};
export default nextConfig;
