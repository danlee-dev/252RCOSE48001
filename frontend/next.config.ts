import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  webpack: (config) => {
    // react-pdf/pdfjs-dist 호환성 설정
    config.resolve.alias.canvas = false;
    return config;
  },
};

export default nextConfig;
